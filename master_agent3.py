import gradio as gr
import json
import os
import sys
import whisper
import subprocess
import importlib.util
import textwrap
import base64
import mimetypes
from datetime import datetime
from litellm import completion
import torch

# ==========================================
# 1. إعدادات البيئة وقاعدة البيانات
# ==========================================
DB_FILE = "agent_vault.json"
SKILLS_DIR = "skills"
os.makedirs(SKILLS_DIR, exist_ok=True)

print("⏳ جاري تحضير \"المخ\" الصوتي (Whisper) على كارت الشاشة...")
device = "cuda" if torch.cuda.is_available() else "cpu"
stt_model = whisper.load_model("base", device=device)

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding="utf-8") as f: 
            return json.load(f)
    return {}

def save_db(db):
    with open(DB_FILE, "w", encoding="utf-8") as f: 
        json.dump(db, f, ensure_ascii=False, indent=4)

# ==========================================
# 2. القاموس واللغات وكتالوج الموديلات
# ==========================================
LANGUAGES = {
    "عربي": {"log": "### 📂 السجل الذكي", "new": "➕ مهمة جديدة", "hist": "المحادثات السابقة", "settings": "🛠️ إعدادات العقل", "prov": "المزود (Provider)", "mod": "الموديل (Model)", "url": "رابط الخادم (Base URL)", "key": "مفتاح الربط (API Key)", "chat": "مركز القيادة", "txt": "الأمر المكتوب (ارفع الصور هنا)", "aud": "الأمر الصوتي", "run": "🚀 تنفيذ", "lang": "🌐 لغة الواجهة"},
    "English": {"log": "### 📂 Smart Log", "new": "➕ New Task", "hist": "Previous Chats", "settings": "🛠️ Mind Settings", "prov": "Provider", "mod": "Model", "url": "Base URL", "key": "API Key", "chat": "Command Center", "txt": "Message (Upload images here)", "aud": "Voice Command", "run": "🚀 Execute", "lang": "🌐 UI Language"},
    "Français": {"log": "### 📂 Journal Intelligent", "new": "➕ Nouvelle Tâche", "hist": "Discussions Précédentes", "settings": "🛠️ Paramètres du Cerveau", "prov": "Fournisseur", "mod": "Modèle", "url": "URL du Serveur", "key": "Clé API", "chat": "Centre de Commande", "txt": "Message (Télécharger des images ici)", "aud": "Commande Vocale", "run": "🚀 Exécuter", "lang": "🌐 Langue"},
    "Español": {"log": "### 📂 Registro Inteligente", "new": "➕ Nueva Tarea", "hist": "Chats Anteriores", "settings": "🛠️ Ajustes de la Mente", "prov": "Proveedor", "mod": "Modelo", "url": "URL del Servidor", "key": "Clave API", "chat": "Centro de Mando", "txt": "Mensaje (Sube imágenes aquí)", "aud": "Comando de Voz", "run": "🚀 Ejecutar", "lang": "🌐 Idioma"},
    "Deutsch": {"log": "### 📂 Smart-Protokoll", "new": "➕ Neue Aufgabe", "hist": "Vorherige Chats", "settings": "🛠️ Gehirneinstellungen", "prov": "Anbieter", "mod": "Modell", "url": "Server-URL", "key": "API-Schlüssel", "chat": "Kommandozentrale", "txt": "Nachricht (Bilder hier hochladen)", "aud": "Sprachbefehl", "run": "🚀 Ausführen", "lang": "🌐 Sprache"},
    "Italiano": {"log": "### 📂 Registro Intelligente", "new": "➕ Nuovo Compito", "hist": "Chat Precedenti", "settings": "🛠️ Impostazioni della Mente", "prov": "Fornitore", "mod": "Modello", "url": "URL del Server", "key": "Chiave API", "chat": "Centro di Comando", "txt": "Messaggio (Carica immagini qui)", "aud": "Comando Vocale", "run": "🚀 Esegui", "lang": "🌐 Lingua"},
    "Русский": {"log": "### 📂 Умный журнал", "new": "➕ Новая задача", "hist": "Предыдущие чаты", "settings": "🛠️ Настройки разума", "prov": "Провайдер", "mod": "Модель", "url": "URL сервера", "key": "API Ключ", "chat": "Командный центр", "txt": "Сообщение (Загрузите изображения здесь)", "aud": "Голосовая команда", "run": "🚀 Выполнить", "lang": "🌐 Язык"},
    "中文": {"log": "### 📂 智能日志", "new": "➕ 新任务", "hist": "历史聊天", "settings": "🛠️ 大脑设置", "prov": "提供商", "mod": "模型", "url": "服务器地址", "key": "API 密钥", "chat": "指挥中心", "txt": "消息 (在此上传图片)", "aud": "语音指令", "run": "🚀 执行", "lang": "🌐 语言"},
    "हिन्दी": {"log": "### 📂 स्मार्ट लॉग", "new": "➕ नया कार्य", "hist": "पिछली चैट", "settings": "🛠️ मस्तिष्क सेटिंग्स", "prov": "प्रदाता", "mod": "मॉडल", "url": "सर्वर URL", "key": "API कुंजी", "chat": "कमांड सेंटर", "txt": "संदेश (यहां चित्र अपलोड करें)", "aud": "ध्वनि आदेश", "run": "🚀 निष्पादित करें", "lang": "🌐 भाषा"}
}

MODELS_CATALOG = {
    "محلي (LM Studio / Ollama / KoboldCPP)": [
        "llama3.1", "llama3", "qwen2.5", "qwen2", "phi-3", "gemma-2", 
        "mistral", "dolphin-mixtral", "llava", "paligemma", "deepseek-coder-v2"
    ],
    "مخصص (Custom API)": [
        "اكتب اسم الموديل الخاص بك هنا..."
    ],
    "Google (Gemini)": [
        "gemini/gemini-1.5-pro", "gemini/gemini-1.5-flash", "gemini/gemini-1.0-pro"
    ],
    "OpenAI": [
        "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo", "o1-preview"
    ],
    "Anthropic (Claude)": [
        "claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-haiku-20240307"
    ],
    "Meta (Llama API / Groq)": [
        "llama-3.1-405b-instruct", "llama-3.1-70b-instruct", "llama3-70b-8192"
    ],
    "NVIDIA": [
        "nemotron-4-340b-instruct", "llama-3-70b-instruct"
    ],
    "Mistral": [
        "mistral-large-latest", "open-mixtral-8x22b"
    ]
}

def update_ui_language(lang):
    t = LANGUAGES.get(lang, LANGUAGES["عربي"])
    return [
        t["log"], gr.update(value=t["new"]), gr.update(label=t["hist"]),
        gr.update(label=t["settings"]), gr.update(label=t["prov"]),
        gr.update(label=t["mod"]), gr.update(label=t["url"]), gr.update(label=t["key"]),
        gr.update(label=t["chat"]), gr.update(label=t["txt"]),
        gr.update(label=t["aud"]), gr.update(value=t["run"]), gr.update(label=t["lang"])
    ]

def update_provider_settings(provider_name):
    available_models = MODELS_CATALOG.get(provider_name, [])
    default_model = available_models[0] if available_models else ""

    if provider_name == "مخصص (Custom API)":
        return gr.update(choices=[], value=""), gr.update(value="", placeholder="http://your-server-ip:port/v1"), gr.update(value="", placeholder="أدخل API Key إذا لزم الأمر")
    elif "محلي" in provider_name:
        return gr.update(choices=available_models, value=default_model), gr.update(value="http://localhost:1234/v1"), gr.update(value="")
    else:
        return gr.update(choices=available_models, value=default_model), gr.update(value="", placeholder="لا تحتاج رابط خادم لهذه الشركة"), gr.update(value="", placeholder="أدخل API Key الخاص بالشركة")

# ==========================================
# 3. نظام المهارات والأدوات الأساسية
# ==========================================
def load_dynamic_tools():
    dynamic_schemas = []
    dynamic_functions = {}
    for filename in os.listdir(SKILLS_DIR):
        if filename.endswith(".py"):
            module_name = filename[:-3]
            spec = importlib.util.spec_from_file_location(module_name, os.path.join(SKILLS_DIR, filename))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, 'TOOL_SCHEMA') and hasattr(module, 'execute'):
                dynamic_schemas.append(module.TOOL_SCHEMA)
                dynamic_functions[module.TOOL_SCHEMA['function']['name']] = module.execute
    return dynamic_schemas, dynamic_functions

def create_new_skill(skill_name, description, python_code, required_packages=[]):
    if required_packages:
        for pkg in required_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            except Exception as e:
                return f"❌ فشل في تسطيب {pkg}. الخطأ: {str(e)}"

    safe_code = textwrap.indent(python_code, '            ')
    filepath = os.path.join(SKILLS_DIR, f"{skill_name}.py")
    file_content = f"""
import io
import contextlib

TOOL_SCHEMA = {{
    "type": "function",
    "function": {{
        "name": "{skill_name}",
        "description": "{description}",
        "parameters": {{"type": "object", "properties": {{"input": {{"type": "string"}}}}}}
    }}
}}

def execute(input):
    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output):
        try:
{safe_code}
        except Exception as e:
            print(f"❌ خطأ بداخل المهارة: {{str(e)}}")
            
    output_str = captured_output.getvalue()
    return output_str if output_str.strip() else "✅ تم تنفيذ المهارة بنجاح."
"""
    with open(filepath, "w", encoding="utf-8") as f: 
        f.write(file_content)
    return f"🛠️ تم بناء المهارة: {skill_name} بنجاح."

def system_control(command):
    try:
        res = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT, timeout=3600)
        return f"✅ تم التنفيذ بنجاح. النتيجة:\n{res}"
    except subprocess.TimeoutExpired:
        return f"❌ خطأ: استغرق الأمر وقتاً طويلاً جداً (أكثر من ساعة) وتم إيقافه."
    except Exception as e: 
        return f"❌ فشل تنفيذ الأمر. الخطأ:\n{str(e)}"

BASE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "system_control",
            "description": "تنفيذ أوامر CMD على نظام الويندوز. استخدمها لتثبيت البرامج أو إدارة الملفات.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_new_skill",
            "description": "برمجة مهارة جديدة لنفسك بـ Python.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "skill_name": {"type": "string"},
                    "description": {"type": "string"},
                    "python_code": {"type": "string"},
                    "required_packages": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["skill_name", "description", "python_code"]
            }
        }
    }
]

# ==========================================
# 4. المعالجة والذكاء (The Brain Logic)
# ==========================================

def get_ui_history(history):
    ui_hist = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        display_text = ""
        
        if "ui_files" in msg and msg["ui_files"]:
            for filepath in msg["ui_files"]:
                try:
                    mime_type, _ = mimetypes.guess_type(filepath)
                    mime_type = mime_type or "image/jpeg"
                    with open(filepath, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                        display_text += f"![الصورة المرفوعة](data:{mime_type};base64,{base64_image})\n\n"
                except Exception as e:
                    print(f"Error loading image: {e}")
        
        if isinstance(content, list):
            text_parts = [c["text"] for c in content if c["type"] == "text"]
            if text_parts:
                display_text += " ".join(text_parts)
        elif isinstance(content, str) and content.strip():
            display_text += content
            
        ui_hist.append({"role": role, "content": display_text})
            
    return ui_hist

def transcribe_voice(audio_path, current_msg_dict):
    if audio_path is None: return current_msg_dict
    text = stt_model.transcribe(audio_path, language="ar")["text"]
    current_msg_dict["text"] = text
    return current_msg_dict

def chat_engine(current_title, msg_dict, provider, model, api_key, url, history, current_lang):
    user_input = msg_dict.get("text", "")
    uploaded_files = msg_dict.get("files", [])
    
    db = load_db()
    if not user_input and not uploaded_files: 
        return current_title, get_ui_history(history), history, gr.update()
    
    if not current_title or current_title == "محادثة جديدة" or current_title == "New Task":
        title_text = user_input[:20] if user_input else "تحليل صورة"
        current_title = f"{title_text}... ({datetime.now().strftime('%H:%M')})"

    content = []
    if user_input:
        content.append({"type": "text", "text": user_input})
        
    if uploaded_files:
        for file_path in uploaded_files:
            mime_type, _ = mimetypes.guess_type(file_path)
            mime_type = mime_type or "image/jpeg"
            with open(file_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                })

    final_content = content if uploaded_files else user_input
    
    if uploaded_files:
        history.append({"role": "user", "content": final_content, "ui_files": uploaded_files})
    else:
        history.append({"role": "user", "content": final_content})
    
    system_instruction = {
        "role": "system", 
        "content": f"""أنت Jarvis، نظام ذكاء اصطناعي مستقل. يجب أن ترد باللغة {current_lang}.
قواعد صارمة:
1. لا تكذب أبداً (No Hallucination).
2. تثبيت البرامج: winget install --id [اسم_البرنامج] -e --source winget --accept-package-agreements --accept-source-agreements --silent
3. لفتح برنامج: start [اسم_البرنامج]
"""
    }
    
    messages_for_llm = [system_instruction] + history
    dyn_schemas, dyn_funcs = load_dynamic_tools()
    all_tools = BASE_TOOLS + dyn_schemas

    if "محلي" in provider or "مخصص" in provider:
        actual_model = f"openai/{model}" 
    else:
        actual_model = model

    try:
        # 🌟 هنا تم تأمين الرابط والمفتاح ضد أخطاء NoneType 🌟
        safe_url = url.strip() if (url and url.strip() != "") else None
        safe_key = api_key.strip() if (api_key and api_key.strip() != "") else "empty"

        response = completion(
            model=actual_model,
            messages=messages_for_llm,
            api_base=safe_url,
            api_key=safe_key,
            tools=all_tools
        )
        
        msg = response.choices[0].message
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            results = []
            for call in msg.tool_calls:
                f_name = call.function.name
                args = json.loads(call.function.arguments) if call.function.arguments else {}
                
                if f_name == "system_control": res = system_control(args.get('command', ''))
                elif f_name == "create_new_skill": res = create_new_skill(args.get('skill_name', ''), args.get('description', ''), args.get('python_code', ''), args.get('required_packages', []))
                elif f_name in dyn_funcs: res = dyn_funcs[f_name](args.get('input', ''))
                
                results.append(f"⚙️ نتيجة الأداة:\n{res}")
            bot_res = "\n\n".join(results)
        else:
            bot_res = msg.content

        history.append({"role": "assistant", "content": bot_res})
        db[current_title] = history
        save_db(db)
        
        return current_title, get_ui_history(history), history, gr.update(choices=list(db.keys()), value=current_title)
    
    except Exception as e:
        error_msg = str(e)
        if "does not support images" in error_msg or "vision" in error_msg.lower():
            nice_error = "👁️❌ عذراً يا سيدي، الموديل الذي اخترته (مثل نماذج Gemma النصية) لا يدعم تحليل الصور. يرجى اختيار موديل يدعم الرؤية البصرية (مثل PaliGemma، LLaVA، أو GPT-4o) وإعادة المحاولة."
        else:
            nice_error = f"❌ خطأ في الاتصال: {error_msg}"
            
        temp_history = history.copy()
        temp_history.append({"role": "assistant", "content": nice_error})
        return current_title, get_ui_history(temp_history), history, gr.update()

def load_previous_chat(selected_title):
    db = load_db()
    hist = db.get(selected_title, [])
    return selected_title, get_ui_history(hist), hist

def start_new_chat():
    return "محادثة جديدة", [], []

# ==========================================
# 5. الواجهة الرسومية (The Interface)
# ==========================================
with gr.Blocks(title="AI OS - Jarvis") as app:
    mem_hist = gr.State([])
    mem_title = gr.State("محادثة جديدة")
    
    with gr.Row():
        with gr.Column(scale=1, variant="panel"):
            lang_drop = gr.Dropdown(choices=list(LANGUAGES.keys()), value="عربي", label="🌐 لغة الواجهة")
            
            ui_log = gr.Markdown("### 📂 السجل الذكي")
            new_btn = gr.Button("➕ مهمة جديدة", variant="secondary")
            history_list = gr.Dropdown(choices=list(load_db().keys()), label="المحادثات السابقة")
            
            with gr.Accordion("🛠️ إعدادات العقل", open=False) as settings_acc:
                with gr.Row():
                    prov = gr.Dropdown(choices=list(MODELS_CATALOG.keys()), label="المزود (Provider)", value="محلي (LM Studio / Ollama / KoboldCPP)")
                    mod = gr.Dropdown(choices=MODELS_CATALOG["محلي (LM Studio / Ollama / KoboldCPP)"], label="الموديل (Model)", value="llama3", allow_custom_value=True)
                
                base_url = gr.Textbox(label="رابط الخادم (Base URL)", value="http://localhost:1234/v1")
                api_key = gr.Textbox(label="مفتاح الربط (API Key)", type="password")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="مركز القيادة", height=500)
            
            with gr.Row():
                txt_in = gr.MultimodalTextbox(label="الأمر المكتوب (ارفع الصور هنا)", file_count="multiple", scale=4)
                aud_in = gr.Audio(label="الأمر الصوتي", sources="microphone", type="filepath", scale=1)
                
            with gr.Row():
                down_btn = gr.DownloadButton("💾 Save As")
                run_btn = gr.Button("🚀 تنفيذ", variant="primary")

    # ---------------------------
    # الربط البرمجي (Event Listeners)
    # ---------------------------
    
    prov.change(fn=update_provider_settings, inputs=[prov], outputs=[mod, base_url, api_key])

    lang_drop.change(
        fn=update_ui_language,
        inputs=[lang_drop],
        outputs=[ui_log, new_btn, history_list, settings_acc, prov, mod, base_url, api_key, chatbot, txt_in, aud_in, run_btn, lang_drop]
    )

    aud_in.change(fn=transcribe_voice, inputs=[aud_in, txt_in], outputs=txt_in)
    
    run_btn.click(
        fn=chat_engine,
        inputs=[mem_title, txt_in, prov, mod, api_key, base_url, mem_hist, lang_drop],
        outputs=[mem_title, chatbot, mem_hist, history_list]
    ).then(
        fn=lambda: {"text": "", "files": []}, inputs=None, outputs=txt_in
    )
    
    txt_in.submit(
        fn=chat_engine,
        inputs=[mem_title, txt_in, prov, mod, api_key, base_url, mem_hist, lang_drop],
        outputs=[mem_title, chatbot, mem_hist, history_list]
    ).then(
        fn=lambda: {"text": "", "files": []}, inputs=None, outputs=txt_in
    )
    
    history_list.change(fn=load_previous_chat, inputs=[history_list], outputs=[mem_title, chatbot, mem_hist])
    new_btn.click(fn=start_new_chat, inputs=[], outputs=[mem_title, chatbot, mem_hist])

if __name__ == "__main__":
    app.launch(inbrowser=True, theme=gr.themes.Monochrome())