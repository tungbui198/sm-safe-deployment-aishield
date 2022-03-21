#!/bin/sh

image=$1

docker run -d --name ${image} --rm -v $(pwd)/test_dir:/opt/ml -p 8080:8080 ${image} serve
