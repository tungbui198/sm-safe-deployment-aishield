# docker cmd to build lambda containers

docker build -t lambda01 -f container/Dockerfile.lambda.local ./container

docker run -p 9000:8080 lambda01

curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d "{\"first_name\": \"tung\", \"last_name\":\"dao\"}"

curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d "{\"audio_filename\": \"tung\", \"audio_base64\":\"dao\"}"

curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d @./payload -H "Content-Type: application/json"


aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin 852039983533.dkr.ecr.ap-southeast-1.amazonaws.com

docker build -t lambda-top3 -f container/Dockerfile.lambda.ecr ./container

docker tag lambda-top3:latest 852039983533.dkr.ecr.ap-southeast-1.amazonaws.com/lambda-top3:latest

docker push 852039983533.dkr.ecr.ap-southeast-1.amazonaws.com/lambda-top3:latest

docker run -it lambda-top3 /bin/bash
