#!/bin/bash
# Copyright 2021 Cylynx
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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