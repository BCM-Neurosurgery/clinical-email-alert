"""
Send a survey to a specified patient
"""

from datetime import datetime, timezone
import json
import requests

with open("/home/settings/DBSPsych-56119/qualtrics-ema-config.json", "r") as file:
    config = json.load(file)

"""
Requires json config file with API Token 'token', dict of patient lookup IDs 'patientid_dict', dict of survey IDs 'surveyid_dict', and a mailing list ID 'mailinglistid'.
"""

token = config["token"]  # api token
patient_ids = config["patient_ids"]  # dict of patients and qualtrics IDs
survey_ids = config["survey_ids"]  # dict of survey IDs
mailinglist_ids = config[
    "mailinglist_ids"
]  # dict of mailing list IDs (need mailing list ID and contact ID)
mailinglistid = mailinglist_ids["OCD"]


def send_survey(patient_id, survey="ISS"):
    """
    Sends a specified patient a (ISS by default) survey over SMS and email immediately.
    """

    ### EMAIL SURVEY:
    # Define the API URL
    url = "https://iad1.qualtrics.com/API/v3/distributions"

    # Define the headers
    headers = {"Content-Type": "application/json", "X-API-TOKEN": token}

    # Define the payload (data)
    payload = {
        "message": {
            # "messageId": "MS_0Vdgn7nLGSQBlYN", # for a preset message in Qualtrics
            "messageText": "From Baylor Neurosurgery, please fill out this survey as soon as you can: ${l://SurveyURL}"
        },
        "recipients": {
            "mailingListId": mailinglistid,  # can specify a group/list or a single contact
            "contactId": patient_ids[patient_id],
        },
        "header": {
            "fromEmail": "u242046@bcm.edu",  # Thomas Baylor email
            "replyToEmail": "u242046@bcm.edu",  # Thomas Baylor email
            "fromName": "Baylor Neurosurgery",
            "subject": "Clinical Survey: Please fill out as soon as you can - "
            + datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M"
            ),  # add timestamp to message otherwise it filters out duplicates
        },
        "surveyLink": {
            "surveyId": survey_ids[survey],
            # "expirationDate": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'), # should we have an expiration date
            "type": "Individual",
        },
        "sendDate": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    # Make the POST request
    response = requests.post(url, headers=headers, json=payload)

    # Print the response
    print(
        "Email response: ", response.json()
    )  # Assuming the response is in JSON format

    ### SMS SURVEY
    # Define the API URL
    url = "https://iad1.qualtrics.com/API/v3/distributions/sms"

    # Define the headers
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-API-TOKEN": token,  # Replace with your actual API token
    }

    # Define the payload (data)
    payload = {
        "message": {
            # "libraryId": "UR_1M4aHozEkSxUfCl",
            # "messageId": "MS_0Vdgn7nLGSQBlYN", # can either load preset messageID or include text directly
            "messageText": "From Baylor Neurosurgery ("
            + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
            + "), please fill out this survey as soon as you can: ${l://SurveyURL}"  # add timestamp to message otherwise it filters out duplicates
        },
        "recipients": {
            "mailingListId": mailinglistid,
            "contactId": patient_ids[patient_id],
        },
        "surveyId": survey_ids[survey],
        "sendDate": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": "Invite",
        "name": "SMS API Trigger",
    }

    # Make the POST request
    response = requests.post(url, headers=headers, json=payload)

    # Print the response
    print("SMS response: ", response.json())
