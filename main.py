import tkinter as tk
from PyPDF2 import PdfReader
import openai
from io import BytesIO
import pygame
from functools import partial
import concurrent.futures
from dotenv import dotenv_values
import threading
import boto3
from typing import Dict, List



class PDFViewer:
    def __init__(self, file_path, aws_access_key_id, aws_secret_access_key, model_name,voice_id='Gregory',output_format='mp3'):
        self.file_path = file_path
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.model_name = model_name
        self.voice_id = voice_id
        self.output_format = output_format
        self.current_page_index = 1  
        self.pdf_reader = PdfReader(file_path)
        self.num_pages = len(self.pdf_reader.pages)

        self.current_audio_index = 0
        self.current_summary_index = 0
        self.current_interp_index = 0
        
        self.read_pdf()
        pygame.init()
        pygame.mixer.init()

    

    def process_page(self,text):
        summary_text_resp = openai.ChatCompletion.create(
            model=model_name,
            messages=self.summarization_prompt_messages(text, 1000),
        )
        summary_text = summary_text_resp['choices'][0]['message']['content']

        if summary_text != "No Content":
            interp_text_resp = openai.ChatCompletion.create(
                model=model_name,
                messages=self.interp_prompt_messages(text, 1000),
            )
            interp_text = interp_text_resp['choices'][0]['message']['content']

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
            future_speech = executor.submit(self.synthesize_text_chunks, text,3000)
            future_summary = executor.submit(self.synthesize_text_chunks, summary_text, 3000)
            future_interp = executor.submit(self.synthesize_text_chunks, interp_text, 3000)

            return future_speech.result(), future_summary.result(), future_interp.result()
        return None, None, None

    def read_pdf(self):
        with open(self.file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            page = self.pdf_reader.pages[self.current_page_index]
            text = page.extract_text() if page else ""
            self.speech, self.summary, self.interp = self.process_page(text)


    def next_page_handler(self):

        if self.current_page_index < self.num_pages:
            self.current_page_index += 1
            threading.Thread(target=self.read_pdf).start()
        else:
            print("No more pages to read.")


    def synthesize_speech(self,text):
        polly = boto3.client('polly', region_name='us-east-1',
                            aws_access_key_id=self.aws_access_key_id,
                            aws_secret_access_key=self.aws_secret_access_key)

        response = polly.synthesize_speech(Engine='long-form', Text=text, OutputFormat=self.output_format, VoiceId=self.voice_id)

        return response['AudioStream'].read()



    def split_text(self,text, max_length):
        return (text[i:i+max_length] for i in range(0, len(text), max_length))

    def synthesize_text_chunks(self,text, max_length):
        audio_chunks = []
        for chunk in self.split_text(text, max_length):
            audio_chunk = self.synthesize_speech(chunk)
            audio_chunks.append(audio_chunk)
        return audio_chunks



    def summarization_prompt_messages(self, text: str, target_summary_size: int) -> List[Dict]:
        return [
            {
                "role": "system",
                "content": f"""
    The user is asking you to summarize a page of a pdf. The pdf will be converted to text to speech.
    Strive to make your summary as detailed as possible while remaining under a {target_summary_size} token limit. If the page does not contain any content to summarize return "No Content"
    """.strip(),
            },
            {"role": "user", "content": f"Summarize the following: {text}"},
        ]

    def interp_prompt_messages(self, text: str, target_summary_size: int) -> List[Dict]:
        return [
            {
                "role": "system",
                "content": f"""
    The user is asking you to interpret a page of a pdf. The pdf will be converted to text to speech.
    Strive to make your interpretation as detailed and as simple as possible while remaining under a {target_summary_size} token limit. 
    """.strip(),
            },
            {"role": "user", "content": f"interpret the following: {text}"},
        ]

    def build_gui(self):
        self.root = tk.Tk()
        self.root.title("Music Player")

        frame = tk.Frame(self.root, pady=20)
        frame.pack()


                # Play button
        play_button = tk.Button(frame, text="Play", command=self.play_speech, padx=10, pady=5, bg="green", fg="white")
        play_button.grid(row=0, column=0)

        # Stop button
        stop_button = tk.Button(frame, text="Stop", command=self.stop_audio, padx=10, pady=5, bg="red", fg="white")
        stop_button.grid(row=0, column=1)

        # Pause button
        pause_button = tk.Button(frame, text="Pause", command=self.pause_audio, padx=10, pady=5, bg="yellow", fg="black")
        pause_button.grid(row=0, column=2)

        # Resume button
        resume_button = tk.Button(frame, text="Resume", command=self.resume_audio, padx=10, pady=5, bg="blue", fg="white")
        resume_button.grid(row=0, column=3)

        # Summary button
        summary_button = tk.Button(frame, text="Summary", command=self.play_summary, padx=10, pady=5, bg="purple", fg="white")
        summary_button.grid(row=0, column=4)

        # Interpretation button
        interpretation_button = tk.Button(frame, text="Interpretation", command=self.play_interp, padx=10, pady=5, bg="cyan", fg="black")
        interpretation_button.grid(row=0, column=5)

        # Next Page button
        next_page_button = tk.Button(frame, text="Next Page", command=self.next_page_handler, padx=10, pady=5, bg="orange", fg="black")
        next_page_button.grid(row=0, column=6)

        self.root.mainloop()

    def play_speech(self):

        if self.current_summary_index < len(self.speech):
            pygame.mixer.music.load(BytesIO(self.speech[self.current_summary_index]))
            pygame.mixer.music.play(loops=0)
            self.current_summary_index += 1
        else:
            self.current_summary_index = 0

    def stop_audio(self):
        pygame.mixer.music.stop()

    def pause_audio(self):
        pygame.mixer.music.pause()
    
    def resume_audio(self):
        pygame.mixer.music.unpause()

    def play_summary(self):
        pass

    def play_interp(self):
        pass


if __name__ == "__main__":
    config = dotenv_values(".env")
    aws_access_key_id = config['AWS_ACCESS_KEY']
    aws_secret_access_key = config['AWS_SECRET_KEY']
    openai.api_key = config['OPENAI_SECRET_KEY']
    model_name = "gpt-3.5-turbo"
    pdf_file_path = "J. Harris.pdf"
    
    viewer = PDFViewer(pdf_file_path, aws_access_key_id, aws_secret_access_key, model_name)
    viewer.build_gui()