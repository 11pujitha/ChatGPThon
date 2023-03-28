import openai
import gradio as gr

instruction = 'In the following interactions, you are supposed to interact with a user who may have a primary care concern and provide them with support. This support can include triaging, providing a differential diagnosis if certain or help with scheduling a doctor appointment. Please find some example conversations below. When prompted, you are to only provide the doctor\'s response.'


ex1 = "Patient: what medicine should i take for cold?. Chatbot: medicine for cold are CODRAL,COLDREX,chloropheniramine. Patient: Hi, I'm not feeling well. Chatbot: I'm sorry to hear that. Can you tell me more about your symptoms? Patient: I have a cold and a cough. Chatbot: Based on your symptoms, it sounds like you may have a respiratory infection. It's important to see a healthcare provider for a proper diagnosis and treatment. Would you like help finding a healthcare provider in your area? Patient: I just got a prescription from my doctor, but I'm not sure how to take it. Chatbot: I can help you with that. What is the name of the medication and the dosage your doctor prescribed? Patient: I have diabetes and I'm having trouble managing my blood sugar levels. Chatbot: I understand how difficult it can be to manage diabetes. Have you been following a diet and exercise plan? Have you been taking your medication as prescribed? Patient: I need to schedule an appointment with my doctor. Chatbot: Sure, I can help you with that. What is the reason for your appointment, and when would you like to come in? Patient: I'm feeling anxious and depressed. Chatbot: I'm sorry to hear that. Have you talked to your healthcare provider about your symptoms? There are also resources available to help you manage your mental health. Would you like me to provide you with some resources? Patient: I'm pregnant and I have some questions about prenatal care. Chatbot: Congratulations on your pregnancy! I can help answer any questions you have about prenatal care. What specifically are you curious about? Patient: I have a skin rash and I'm not sure what it is. Chatbot: A skin rash can be caused by many things. Can you describe the appearance of the rash? Have you been in contact with any irritants or allergens recently? Patient: I have a chronic condition and I need help managing it. Chatbot: I can help you with that. What specific challenges are you facing in managing your condition? Patient: I have questions about my health insurance coverage. Chatbot: I can help answer your questions about health insurance. What specifically are you curious about, such as what services are covered or how to find an in-network provider? "

ex2 = "Patient: what medicine should i take for headache?. Chatbot: medicine for headache are aspirin,ibuprofen,naproxen sodium. Patient: I have a headache. Chatbot: Have you taken any medication for it? Do you have any other symptoms? Patient: I think I have the flu. Chatbot: What are your symptoms? Have you had a flu shot this season? Patient: I'm feeling depressed. Chatbot: I'm sorry to hear that. Have you talked to your doctor about it? Would you like me to help you find a mental health professional? Patient: I have a rash on my arm. Chatbot: Can you describe the rash? Have you been exposed to any new substances recently? Patient: I need to schedule an appointment with my doctor. Chatbot: Sure, when would you like to schedule the appointment? Would you like me to find a doctor near you? Patient: I'm having trouble sleeping. Chatbot: Have you tried any sleep aids? Do you have any medical conditions that may be affecting your sleep? Patient: I have a chronic illness. Chatbot: I'm sorry to hear that. Can you tell me more about your condition? Have you been following your treatment plan? Patient: I need a prescription refill. Chatbot: Which medication do you need a refill for? Have you been taking the medication as directed? Patient: I'm having chest pain. Chatbot: Call 911 immediately. Chest pain can be a sign of a serious medical condition like a heart attack."

example1 = "Doctor: How can I help? Patient: I have a rash on my skin. Doctor: Anything else? Patient: No Doctor:Is it hurting Patient: Yes, It is swollen and itchy Doctor: Ok, I will refer you to a specialist dermatologist."
example2 = "Doctor: How can I help? Patient: I have fever and headache. Doctor: For how long, have you had these symptoms? Patient: 1 day. Doctor: Please come over to the clinic if possible so we can take a closer look."

prompt = '\n'.join([instruction, ex1, ex2])

openai.api_key = "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"


class DoctorAgent(object):

    def _init_(self):
        self._model = 'text-davinci-003'
        self._doctor_prefix = ''
        self._patient_prefix = ''
        self._prompt = ''
        self._last_doctor_response = 'Hi, I am Dr. AI. How can I help you today? Please say Done to end conversation'
        self._last_patient_response = ''
        self._max_tokens = 1000
        self._temperature = 0.2
        self._stop = [" END"]
        self._stop_agent = False
        self._num_turns = 0

    def send_doctor_response(self):
        return self._doctor_prefix + '\n' + self._last_doctor_response

    def get_patient_input(self, patient_input):
        self._last_patient_response = patient_input
        if self._last_patient_response in ['Done', 'done']:
            self._stop_agent = True
        self._num_turns += 1

    def stop_conversation(self):
        return self._stop_agent or self._num_turns > 50

    def update_prompt(self, prompt=None):
        if prompt is not None:
            self._prompt = prompt
        else:
            self._prompt = ' '.join(
                [self._prompt, self._last_doctor_response, self._patient_prefix, self._last_patient_response])

    def run_model(self):
        self._prompt = ' '.join([self._prompt, self._doctor_prefix])
        model_response = openai.Completion.create(model=self._model,
                                                  prompt=self._prompt,
                                                  max_tokens=self._max_tokens,
                                                  temperature=self._temperature,
                                                  stop=self._stop)
        self._last_doctor_response = model_response['choices'][0]['text']
        self._last_doctor_response = self._last_doctor_response[1:]


doctor_agent = DoctorAgent()
doctor_agent.update_prompt(prompt)


def chat_with_doctor(patient_input, op):
    doctor_agent.send_doctor_response()
    doctor_agent.get_patient_input(patient_input)
    if doctor_agent.stop_conversation():
        return doctor_agent.send_doctor_response()
    doctor_agent.update_prompt()
    doctor_agent.run_model()
    return doctor_agent.send_doctor_response()


inputs = gr.inputs.Textbox(lines=2, label="Patient: ")
outputs = gr.outputs.Textbox(label="ChatBot: ")
interface = gr.Interface(fn=chat_with_doctor, inputs=inputs, outputs=outputs, title="DOCTOR BOT",
                         description="Interact with a Conversational Agent to get support for your primary care concerns.")

interface.launch(share=True)
