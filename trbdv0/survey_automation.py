"""
Send a Qualtrics survey/message to a specified patient via email and/or SMS
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
patient_ids = config["lookup_ids"]  # dict of patients and qualtrics IDs
survey_ids = config["survey_ids"]  # dict of survey IDs
mailinglist_id = config[
    "mailinglist_id"
]  # dict of mailing list IDs (need mailing list ID and contact ID)
# message_ids = config['message_ids']

messageText = (
    "From Baylor Research ("
    + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    + "), please fill out this survey as soon as you can: ${l://SurveyURL}"
)


def send_survey(patient_id, survey="ISS"):
    """
    Sends a specified patient a survey (ISS by default) or reminder over SMS and email immediately.
    """
    if patient_id[0] == "T":
        mailinglistid = mailinglist_id["TRBD"]
    elif patient_id[0] == "D":
        mailinglistid = mailinglist_id["OCD"]

    ### EMAIL SURVEY:
    # Define the API URL
    url = "https://iad1.qualtrics.com/API/v3/distributions"

    # Define the headers
    headers = {"Content-Type": "application/json", "X-API-TOKEN": token}

    # Define the payload (data)
    payload = {
        "message": {
            # "messageId": message_ids['email_survey'] # for a preset message in Qualtrics
            "messageText": messageText
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
        "surveyLink": {"surveyId": survey_ids[survey], "type": "Individual"},
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
            # "messageId": message_ids['email_survey'], # can either load preset messageID or include text directly
            "messageText": messageText  # add timestamp to message otherwise it filters out duplicates
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


def send_wearable_reminder(patient_id):
    """
    Sends a specified patient a reminder to wear their Oura Ring by email and SMS
    """
    if patient_id[0] == "T":
        mailinglistid = mailinglist_id["TRBD"]
    else:
        mailinglistid = mailinglist_id["OCD"]

    ### EMAIL SURVEY:
    # Define the API URL
    url = "https://iad1.qualtrics.com/API/v3/distributions"

    # Define the headers
    headers = {"Content-Type": "application/json", "X-API-TOKEN": token}

    # Define the payload (data)
    payload = {
        "message": {
            # "libraryId": library_id,
            # "messageId": message_ids['wearable_reminder_email'], # for a preset message in Qualtrics
            "messageText": "From Baylor Neurosurgery ("
            + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
            + "), we hope you are doing well. We noticed a lack of Oura Ring data coming in, please fill out this survey if there is anything you would like to tell us: ${l://SurveyURL}"
        },
        "recipients": {
            "mailingListId": mailinglistid,  # can specify a group/list or a single contact
            "contactId": patient_ids[patient_id],
        },
        "header": {
            "fromEmail": "u242046@bcm.edu",  # Thomas Baylor email
            "replyToEmail": "u242046@bcm.edu",  # Thomas Baylor email
            "fromName": "Baylor Neurosurgery",
            "subject": "Lack of wearable data coming in - "
            + datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M"
            ),  # add timestamp to message otherwise it filters out duplicates
        },
        "surveyLink": {
            "surveyId": survey_ids["Short_Response"],
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
            # "libraryId": library_id,
            # "messageId": message_ids['wearable_reminder_sms'], # can either load preset messageID or include text directly
            "messageText": "From Baylor Neurosurgery ("
            + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
            + "), we hope you are doing well. We noticed a lack of Oura Ring data coming in, please fill out this survey if there is anything we should know: ${l://SurveyURL}"  # add timestamp to message otherwise it filters out duplicates
        },
        "recipients": {
            "mailingListId": mailinglistid,
            "contactId": patient_ids[patient_id],
        },
        "surveyId": survey_ids["Short_Response"],
        "sendDate": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": "Invite",
        "name": "SMS API Trigger",
    }

    # Make the POST request
    response = requests.post(url, headers=headers, json=payload)

    # Print the response
    print("SMS response: ", response.json())
