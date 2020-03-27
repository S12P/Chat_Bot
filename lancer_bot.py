from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging, os

# from rasa_core.agent import Agent
# # from rasa_core.channels.console import ConsoleInputChannel
# from rasa_core.interpreter import RegexInterpreter
# from rasa_core.policies.keras_policy import KerasPolicy
# from rasa_core.policies.memoization import MemoizationPolicy
# from rasa_core.interpreter import RasaNLUInterpreter

from rasa_core.channels.rasa_chat import RasaChatInput
from rasa_core.channels.channel import CollectingOutputChannel, UserMessage
from rasa_core.agent import Agent
from rasa_core.interpreter import RasaNLUInterpreter
from rasa_core.utils import EndpointConfig
from rasa_core import utils

from flask import render_template, Blueprint, jsonify, request

import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

logger = logging.getLogger(__name__)

faq_data = pd.read_csv("./data/faq_data.csv")

def train_dialogue(domain_file = 'faq_domain.yml',
                    model_path = './models/dialogue',
                    training_data_file = 'data/stories.md'):

    agent = Agent(domain_file, policies = [MemoizationPolicy(), KerasPolicy()])

    agent.train(
                training_data_file,
                epochs = 300,
                batch_size = 50,
                validation_split = 0.2)

    agent.persist(model_path)
    return agent

interpreter = RasaNLUInterpreter('./models/nlu/default/faq_bot')
MODEL_PATH = "models/dialogue"

#action_endpoint = EndpointConfig(url="https://localhost:5050/rasa/webhook")
#agent = Agent.load(MODEL_PATH, interpreter=interpreter, action_endpoint=action_endpoint)
agent = Agent.load(MODEL_PATH, interpreter=interpreter)


class MyNewInput(RasaChatInput):
    @classmethod
    def name(cls):
        return "rasa"
    def _check_token(self, token):
        if token == 'mysecret':
            return {'username': 1234}
        else:
            print("Failed to check token: {}.".format(token))
            return None
    def blueprint(self, on_new_message):
        templates_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.join('centraleprojet', 'templates'))
        static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.join('centraleprojet', 'static'))
        custom_webhook = Blueprint('custom_webhook', __name__, template_folder=templates_folder, static_folder=static_folder, root_path=__name__)
        interpreter = RasaNLUInterpreter('./models/nlu/default/faq_bot')
        MODEL_PATH = "./models/dialogue"
        faq_data = pd.read_csv("./data/faq_data.csv")
        @custom_webhook.route("/", methods=['GET'])
        def health():
            return jsonify({"status": "ok"})
        @custom_webhook.route("/chat", methods=['GET'])
        def chat():
            return render_template('index.html')
        @custom_webhook.route("/webhook", methods=['POST'])
        def receive():
            agent = Agent.load(MODEL_PATH, interpreter=interpreter)
            sender_id = self._extract_sender(request)
            text = self._extract_message(request)
            aht = agent.handle_text(text)
            answer = ""
            if aht == []:
                questions = faq_data['question'].values.tolist()
                mathed_question, score = process.extractOne(text, questions, scorer=fuzz.token_set_ratio) # use process.extract(.. limits = 3) to get multiple close matches
                if score > 50:
                    matched_row = faq_data.loc[faq_data['question'] == mathed_question,]
                    match = matched_row['question'].values[0]
                    answer = matched_row['answers'].values[0]
                else:
                    answer = "Sorry I didn't find anything relevant to your query!"
            else:
                answer = aht[0]['text']
            #should_use_stream = utils.bool_arg("stream", default=False)
            '''if should_use_stream:
                return Response(self.stream_response(on_new_message, text, sender_id), content_type='text/event-stream')
            else:
                collector = CollectingOutputChannel()
                print("Aquí")
                on_new_message(UserMessage(text, collector, sender_id))
                print ("json", collector.messages)
                return jsonify(collector.messages)'''
            reponse = \
                        [{
                            "recipient_id" : sender_id,
                            "text" : answer
                        }]
            return jsonify(reponse)
        return custom_webhook

#input_channel = MyNewInput(url='https://chatbot-ecl-manage.herokuapp.com/')
input_channel = MyNewInput(url='centraleprojet.herokuapp.com/')
# set serve_forever=False if you want to keep the server running
s = agent.handle_channels([input_channel],  int(os.environ.get('PORT', 5004)), serve_forever=True)

'''
agent.is_ready()
while(True):
    reponse = ""
    query = input("Votre message :")
    aht = agent.handle_text(query)
    if aht == []:
        questions = faq_data['question'].values.tolist()
        mathed_question, score = process.extractOne(query, questions, scorer=fuzz.token_set_ratio) # use process.extract(.. limits = 3) to get multiple close matches
        if score > 50:
            matched_row = faq_data.loc[faq_data['question'] == mathed_question,]
            match = matched_row['question'].values[0]
            answer = matched_row['answers'].values[0]
            reponse = "\n\n Question: {} \n Answer: {} \n".format(match, answer)
        else:
            reponse = "Sorry I didn't find anything relevant to your query!"
    else:
        reponse = aht[0]['text']
    print(reponse)
'''