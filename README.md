# GeminiAPIMyBotDiscord
My discord bot on gemini api with models selection<br>
There is api_keys.json file for storing your keys for discord and gemini api<br>
You can change something like system prompt in DEFAULT_SYSTEM_PROMPT and also set your own prompt by using prompt command. 
Keep in mind you can send it videos and music too!!
available commands:
"/model" available values 
MODELS = {
    '2.5 flash latest': 'gemini-2.5-flash-lite-preview-09-2025',
    '2.5 flash lite latest': 'gemini-2.5-flash-preview-09-2025',
    '2.0 flash': 'gemini-2.0-flash',
    '2.0 flash lite': 'gemini-2.0-flash-lite',
    '2.5 flash lite': 'gemini-2.5-flash-lite',
    '2.5 flash': 'gemini-2.5-flash',
    '2.5 pro': 'gemini-2.5-pro',
    '2.5 pro preview': 'gemini-2.5-pro-preview-06-05',
    '2.5 pro exp': 'gemini-2.5-pro-exp-03-25',
    '2.5 flash preview': 'gemini-2.5-flash-preview-05-20',
    '2.0 pro exp': 'gemini-2.0-pro-exp-02-05',
    '2.0 flash thinking exp': 'gemini-2.0-flash-thinking-exp-01-21',
    '2.0 flash lite preview': 'gemini-2.0-flash-lite-preview-02-05'
}
"/activate" make your bot reply to every message
"/deactivate" opposite of previous one
"/reset" clear history and user prompt
"/prompt" set prompt over system prompt
"/pro_model_limit" just a tokens limit for pro model, in other models works by default but on pro its more cut value (could be set in file ofc, i put about 30k, but the max per minute is 125k, it just gives 429 error while other models dont in a meantime when your history raises so this should be enable when it reaches its limit per minute) 
That should be all! Some stuff on russian including comments but by default it talks english anyway.
