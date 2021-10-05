#!/bin/bash -xue
#
# Uses https://github.com/chrusty/protoc-gen-jsonschema to generate the json schema
# Assumes there is an existing model_card.proto

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
parent_dir="$(dirname "$parent_path")"

VERSION=v0.0.3
OUT_PATH=$parent_dir/schema/$VERSION

mkdir -p $OUT_PATH

# Restrict to only the main ModelCard message
protoc \
--jsonschema_out=messages=[ModelCard]:$OUT_PATH \
--proto_path=$parent_dir/proto \
$parent_dir/proto/model_card.proto

# rename ModelCard.json to model_card.schema.json
mv $OUT_PATH/ModelCard.json $OUT_PATH/model_card.schema.json