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

from mailersend import emails
import json

mail_body = {}

mail_from = {
    "name": "Timothy Lin",
    "email": "hello@cylynx.io",
}


def tally_form_email(request):
    """Forward the response of the FEAT tally form to the user's email"""

    request_json = request.get_json()
    questions = request_json["data"]["fields"]

    match = next(filter(lambda d: d["label"] == "Please enter your email", questions))
    if match["type"] in ["INPUT_EMAIL"]:
        recipient_email = match["value"]
    else:
        raise ValueError("No valid email input field detected")
    recipients = [
        {
            "name": recipient_email.split("@")[0],
            "email": recipient_email,
        }
    ]

    mailer = emails.NewEmail()
    mailer.set_mail_from(mail_from, mail_body)
    mailer.set_mail_to(recipients, mail_body)
    mailer.set_subject("FEAT form response", mail_body)
    mailer.set_plaintext_content(
        f"""
Here's a copy of your response in json format. 
Copy the section after '---' and save it as a json file.
You can then use the tally_form function in our toolkit to bootstrap a model card.
---
{json.dumps(request_json)}
""",
        mail_body,
    )

    send = mailer.send(mail_body)

    return send
