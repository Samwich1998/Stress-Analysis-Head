import json
import random


class simulatedEmotion:

    @staticmethod
    def generate_random_demographics():
        responses = {}
        questionnaire_path = "therapyHelperFiles/surveyInformation/Subject Demographics.json"
        with open(questionnaire_path, 'r') as file:
            survey_questions = json.load(file)
        for key, value in survey_questions["surveyQuestions"].items():
            if "Dropdown" in value:
                responses[key] = random.choice(value["Dropdown"][0])
            if key == "Age":
                responses[key] = random.randint(20, 40)
            elif key == "Weight":
                responses[key] = random.randint(100, 250)
            elif key == "Height":
                ft = random.randint(5, 6)
                inch = random.randint(0, 11)
                responses["Height (ft)"] = ft
                responses["Height (in)"] = inch
        print(responses)
        return responses

    @staticmethod
    def generate_random_emotion_profile():
        random_stai_score = random.randint(20, 80)
        random_emotion_profile = []

        questionnaire_path = "therapyHelperFiles/surveyInformation/I-STAI-Y1 Questions.json"
        with open(questionnaire_path, 'r') as file:
            survey_questions = json.load(file)
            answers = [random.choice(survey_questions["answerChoices"]) for _ in survey_questions["questions"]]
            random_emotion_profile.append(", ".join([f"{q} {a}" for q, a in zip(survey_questions["questions"], answers)]))

        questionnaire_path = "therapyHelperFiles/surveyInformation/PANAS Questions.json"
        with open(questionnaire_path, 'r') as file:
            survey_questions = json.load(file)
            answers = [random.choice(survey_questions["answerChoices"]) for _ in survey_questions["questions"]]
            random_emotion_profile.append(", ".join([f"{q} {a}" for q, a in zip(survey_questions["questions"], answers)]))

        return random_stai_score, random_emotion_profile
