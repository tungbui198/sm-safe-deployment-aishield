import asyncio
import base64
import json
import logging
import os
import time
from pathlib import Path

import aiofiles
import boto3
from botocore.exceptions import ClientError

TEMP_DIR = Path("/tmp")
# Fix write permission for librosa
# https://github.com/librosa/librosa/issues/1156#issuecomment-714381149
os.environ["NUMBA_CACHE_DIR"] = TEMP_DIR.as_posix()
from utils import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
logger.info(f"ENDPOINT_NAME {ENDPOINT_NAME}")
ASSETS_DIR = Path(os.environ["ASSETS_DIR"])
SCALERS_DIR = ASSETS_DIR / "scalers"
logger.info(f"ASSETS_DIR {ASSETS_DIR}")

sm_runtime = boto3.client("runtime.sagemaker")
region = boto3.Session().region_name
boto_session = boto3.Session(region_name=region)


def format_results(results):
    return list(map(float, results.split("\n")[:-1]))


async def save_audio_file(audio_path, audio_base64):
    logger.info(f"save_audio_file")
    async with aiofiles.open(audio_path, "wb") as f:
        content = base64.b64decode(audio_base64)
        await f.write(content)  # async write


def await_save_audio_file(audio_filename, audio_base64):
    logger.info(f"await_save_audio_file")
    timestamp = time.time()
    ext = audio_filename.split(".")[-1]
    audio_path = TEMP_DIR / f"{audio_filename}-{timestamp}.{ext}"

    loop = asyncio.get_event_loop()
    loop.run_until_complete(save_audio_file(audio_path.as_posix(), audio_base64))

    if ext != "wav":
        try:
            logger.info(f"convert_to_wav")
            audio_path = convert_to_wav(audio_path)
        except:
            raise Exception("The API doesn't support {} files.".format(ext))
    return audio_path


def get_payload_from_audio(audio_path):
    logger.info(f"get_payload")
    files = list_dir(SCALERS_DIR)
    logger.info(f"SCALERS_DIR: {files}")

    mfcc, chroma, mel, zcr = mean_feature(audio_path)
    zcr = get_scaler(zcr, SCALERS_DIR, "zcr").reshape(-1)
    mfcc = get_scaler(mfcc, SCALERS_DIR, "mfcc").reshape(-1)
    chroma = get_scaler(chroma, SCALERS_DIR, "chroma").reshape(-1)
    mel = get_scaler(mel, SCALERS_DIR, "mel").reshape(-1)
    x: np.ndarray = np.concatenate([zcr, mfcc, chroma, mel], axis=0).reshape(-1)
    payload = ",".join(x.astype(str))
    return payload


def format_response(message, status_code, content_type):
    return {
        "statusCode": str(status_code),
        "body": json.dumps(message),
        "headers": {
            "Content-Type": content_type,
            "Access-Control-Allow-Origin": "*",
            "X-SageMaker-Endpoint": ENDPOINT_NAME,
        },
    }


def get_prediction(event):
    logger.info(f"get_prediction")
    content_type = event["headers"].get("Content-Type", "text/csv")
    custom_attributes = event["headers"].get("X-Amzn-SageMaker-Custom-Attributes", "")
    logger.info(f"content_type: {content_type}")
    logger.info(f"custom_attributes: {custom_attributes}")

    orig_payload = event["body"]
    if content_type.startswith("application/json"):
        orig_payload = json.loads(orig_payload)
        audio_filename = orig_payload["audio_filename"]
        audio_base64 = orig_payload["audio_base64"]
        audio_path = await_save_audio_file(audio_filename, audio_base64)
        csv_payload = get_payload_from_audio(audio_path)
    elif content_type.startswith("text/csv"):
        csv_payload = orig_payload
    else:
        message = "bad content type: {}".format(content_type)
        logger.error()
        return format_response({"message": message}, 500)

    logger.info(f"payload len: {len(csv_payload)}")
    response = sm_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        Body=csv_payload,
        ContentType="text/csv",
    )
    results = response["Body"].read().decode()
    preds = format_results(results)
    return format_response({"prediction": preds}, 200, content_type)


def lambda_handler(event, context):
    logger.info("## CONTEXT")
    logger.info(f"Lambda function ARN: {context.invoked_function_arn}")
    logger.info(f"Lambda function memory limits in MB: {context.memory_limit_in_mb}")
    logger.info(f"Time remaining in MS: {context.get_remaining_time_in_millis()}")
    logger.info("## EVENT")
    logger.info(json.dumps(event, indent=2)[:100])
    try:
        response = get_prediction(event)
        logger.info(response)
        return response
    except ClientError as e:
        logger.error(
            "Unexpected sagemaker error: {}".format(e.response["Error"]["Message"])
        )
        logger.error(e)
        content_type = event["headers"].get("Content-Type", "text/csv")
        return format_response(
            {"message": "Unexpected sagemaker error"}, 500, content_type
        )
