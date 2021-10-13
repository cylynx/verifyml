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
# Uses mailersend to send out response. Please set MAILERSEND_API_KEY variable.

gcloud functions deploy tally_form_email \
 --runtime python38 \
 --set-env-vars MAILERSEND_API_KEY=${MAILERSEND_API_KEY} \
 --trigger-http \
 --allow-unauthenticated
