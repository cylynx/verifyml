#!/bin/bash -xue
# 
# Uses mailersend to send out response. Please set MAILERSEND_API_KEY variable.

gcloud functions deploy tally_form_email \
 --runtime python38 \
 --set-env-vars MAILERSEND_API_KEY=${MAILERSEND_API_KEY} \
 --trigger-http \
 --allow-unauthenticated
