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
action_endpoint = EndpointConfig(url="https://localhost:5005/webhook")

agent = Agent.load(MODEL_PATH, interpreter=interpreter, action_endpoint=action_endpoint)

print(agent.is_ready())
while(True):
	query = input("Votre message :")
	print(agent.handle_text(query))

	questions = faq_data['question'].values.tolist()
	mathed_question, score = process.extractOne(query, questions, scorer=fuzz.token_set_ratio) # use process.extract(.. limits = 3) to get multiple close matches
	if score > 50:
		matched_row = faq_data.loc[faq_data['question'] == mathed_question,]
		print("werrgwe",matched_row)
		match = matched_row['question'].values[0]
		answer = matched_row['answers'].values[0]
		print("Here's something I found, \n\n Question: {} \n Answer: {} \n".format(match, answer))
	else:
		print("Sorry I didn't find anything relevant to your query!")
