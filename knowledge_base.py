"""Bilingual (Hindi/English) plant-health knowledge base + intent detection."""
import re

KB = {
    "healthy": {
        "name": {"en": "Healthy leaf", "hi": "स्वस्थ पत्ता"},
        "symptoms": {"en": "Even green colour, no spots, holes or powder.", "hi": "एक समान हरा रंग, कोई धब्बा, छेद या पाउडर नहीं।"},
        "causes": {"en": "No disease signs found.", "hi": "बीमारी के कोई लक्षण नहीं मिले।"},
        "treatment": {"en": "No treatment needed. Water at the base and check leaves weekly.", "hi": "इलाज की ज़रूरत नहीं। जड़ में पानी दें और हर हफ्ते पत्ते जाँचें।"},
        "prevention": {"en": "Give sunlight, spacing and balanced nutrition; remove dead leaves.", "hi": "धूप, उचित दूरी और संतुलित खाद दें; सूखे पत्ते हटाते रहें।"},
        "safety": {"en": "-", "hi": "-"},
    },
    "early_blight": {
        "name": {"en": "Early Blight (Alternaria)", "hi": "अर्ली ब्लाइट (अगेती झुलसा)"},
        "symptoms": {"en": "Dark brown spots with ring-like patterns on older leaves, often with a yellow halo.", "hi": "पुराने पत्तों पर छल्लेदार गहरे भूरे धब्बे, अक्सर चारों ओर पीला घेरा।"},
        "causes": {"en": "Fungus spreading in warm, humid weather via rain splash and infected debris.", "hi": "गर्म और नम मौसम में फफूँद, बारिश की छींटों और संक्रमित अवशेषों से फैलती है।"},
        "treatment": {"en": "Remove infected leaves. Spray a copper-based or mancozeb fungicide as per label every 7-10 days. Neem oil can help in mild cases.", "hi": "संक्रमित पत्ते हटाएँ। लेबल के अनुसार कॉपर-आधारित या मैंकोज़ेब फफूँदनाशक का 7-10 दिन में छिड़काव करें। हल्के मामले में नीम तेल मदद करता है।"},
        "prevention": {"en": "Space plants, water at the base, stake plants, rotate crops yearly and clear fallen debris.", "hi": "पौधों में दूरी रखें, जड़ में पानी दें, सहारा दें, हर साल फसल बदलें और गिरे अवशेष साफ करें।"},
        "safety": {"en": "Wear gloves and a mask while spraying; follow the label dose and pre-harvest waiting period.", "hi": "छिड़काव में दस्ताने और मास्क पहनें; लेबल की मात्रा और कटाई से पहले का प्रतीक्षा समय मानें।"},
    },
    "late_blight": {
        "name": {"en": "Late Blight (Phytophthora)", "hi": "लेट ब्लाइट (पछेती झुलसा)"},
        "symptoms": {"en": "Large dark, water-soaked patches that spread fast; white fuzz underneath in damp weather.", "hi": "बड़े गहरे, पानी-सोखे धब्बे जो तेज़ी से फैलते हैं; नमी में नीचे सफेद रुई जैसी परत।"},
        "causes": {"en": "Water mould favoured by cool, wet, cloudy weather; spreads by wind and rain.", "hi": "ठंडे, गीले, बादल वाले मौसम में पनपने वाली फफूँद; हवा और बारिश से फैलती है।"},
        "treatment": {"en": "Act quickly: remove and destroy infected plants and spray a systemic fungicide as per label. Ask your local agriculture office.", "hi": "जल्दी कदम उठाएँ: संक्रमित पौधे नष्ट करें और लेबल के अनुसार सिस्टमिक फफूँदनाशक छिड़कें। स्थानीय कृषि कार्यालय से पूछें।"},
        "prevention": {"en": "Use resistant varieties, avoid overhead watering, ensure airflow and never compost infected plants.", "hi": "रोग-प्रतिरोधी किस्में लगाएँ, ऊपर से पानी न दें, हवा का आवागमन रखें और संक्रमित पौधों की खाद न बनाएँ।"},
        "safety": {"en": "Use protective gear; keep sprays away from children, pets and water sources.", "hi": "सुरक्षा साधन पहनें; दवा बच्चों, पालतू जानवरों और जल स्रोतों से दूर रखें।"},
    },
    "powdery_mildew": {
        "name": {"en": "Powdery Mildew", "hi": "चूर्णिल आसिता (सफेद पाउडर रोग)"},
        "symptoms": {"en": "White or grey powdery coating on leaves; leaves may curl and yellow.", "hi": "पत्तों पर सफेद या धूसर पाउडर जैसी परत; पत्ते मुड़ या पीले हो सकते हैं।"},
        "causes": {"en": "Fungus favoured by dry days, cool humid nights and crowded plants.", "hi": "सूखे दिन, ठंडी नम रातों और घने पौधों में पनपने वाली फफूँद।"},
        "treatment": {"en": "Remove badly affected leaves. Spray wettable sulphur or a baking-soda solution weekly.", "hi": "ज़्यादा प्रभावित पत्ते हटाएँ। घुलनशील सल्फर या बेकिंग सोडा घोल का साप्ताहिक छिड़काव करें।"},
        "prevention": {"en": "Prune for airflow, avoid excess nitrogen, give morning sun and water at the base.", "hi": "हवा के लिए छँटाई करें, ज़्यादा नाइट्रोजन से बचें, सुबह की धूप दें और जड़ में पानी दें।"},
        "safety": {"en": "Do not spray sulphur above 35 C; it can burn leaves.", "hi": "35 C से ऊपर सल्फर न छिड़कें; पत्ते जल सकते हैं।"},
    },
    "rust": {
        "name": {"en": "Leaf Rust", "hi": "पत्ती रतुआ (रस्ट)"},
        "symptoms": {"en": "Orange-rust coloured pustules on leaves that rub off as powder.", "hi": "पत्तों पर नारंगी-जंग रंग के उभार जो पाउडर की तरह झड़ते हैं।"},
        "causes": {"en": "Rust fungus spread by wind spores in humid conditions.", "hi": "नमी में हवा के बीजाणुओं से फैलने वाली रतुआ फफूँद।"},
        "treatment": {"en": "Pick off infected leaves, then spray propiconazole or a copper fungicide per label.", "hi": "संक्रमित पत्ते तोड़ें, फिर लेबल के अनुसार प्रोपिकोनाज़ोल या कॉपर फफूँदनाशक छिड़कें।"},
        "prevention": {"en": "Avoid wet leaves, space plants well and clear fallen leaves.", "hi": "पत्ते गीले न रखें, पौधों में दूरी रखें और गिरे पत्ते साफ करें।"},
        "safety": {"en": "Wear gloves and a mask; respect the harvest waiting period.", "hi": "दस्ताने और मास्क पहनें; कटाई से पहले का प्रतीक्षा समय मानें।"},
    },
    "unknown": {
        "name": {"en": "Plant disease (details not in knowledge base yet)", "hi": "पौधे की बीमारी (जानकारी अभी ज्ञान आधार में नहीं)"},
        "symptoms": {"en": "Not available yet.", "hi": "अभी उपलब्ध नहीं।"},
        "causes": {"en": "Not available yet.", "hi": "अभी उपलब्ध नहीं।"},
        "treatment": {"en": "Please consult your local agriculture office or Krishi Vigyan Kendra.", "hi": "कृपया अपने स्थानीय कृषि कार्यालय या कृषि विज्ञान केंद्र से सलाह लें।"},
        "prevention": {"en": "Remove infected leaves and keep the plants well spaced.", "hi": "संक्रमित पत्ते हटाएँ और पौधों में दूरी रखें।"},
        "safety": {"en": "Follow product labels carefully.", "hi": "उत्पाद के लेबल का ध्यान से पालन करें।"},
    },
}

INTENTS = [
    ("treatment", re.compile(r"treat|cure|spray|medicine|fungicide|remedy|what (should|do) i do|इलाज|उपचार|दवा|दवाई|छिड़क|क्या करूं|क्या करें|ठीक", re.I)),
    ("prevention", re.compile(r"prevent|avoid|again|future|stop|रोक|बचाव|बचा|दोबारा|फिर से", re.I)),
    ("causes", re.compile(r"why|cause|reason|how did|क्यों|कारण|वजह", re.I)),
    ("safety", re.compile(r"safe|safety|danger|harm|child|pet|सुरक्षित|सुरक्षा|खतरा|नुकसान", re.I)),
    ("symptoms", re.compile(r"symptom|sign|look like|लक्षण|निशान", re.I)),
]


def label_to_key(label: str) -> str:
    """Map a PlantVillage-style label (e.g. 'Tomato___Early_blight') to a KB key."""
    k = label.split("___")[-1].lower()
    if "healthy" in k: return "healthy"
    if "early_blight" in k: return "early_blight"
    if "late_blight" in k: return "late_blight"
    if "powdery" in k: return "powdery_mildew"
    if "rust" in k: return "rust"
    return "unknown"


def detect_intent(question: str) -> str:
    for name, rx in INTENTS:
        if rx.search(question or ""):
            return name
    return "summary"


LABELS = {"treatment": ("Treatment", "उपचार"), "prevention": ("Prevention", "रोकथाम"),
          "causes": ("Causes", "कारण"), "safety": ("Safety", "सुरक्षा"), "symptoms": ("Symptoms", "लक्षण")}


def answer(label: str, question: str = "", lang: str = "en") -> dict:
    lang = "hi" if lang == "hi" else "en"
    e = KB[label_to_key(label)]
    intent = detect_intent(question)
    if intent == "summary":
        text = f"{e['name'][lang]}\n{e['symptoms'][lang]}\n{e['treatment'][lang]}"
    else:
        text = f"{e['name'][lang]} - {LABELS[intent][lang == 'hi']}:\n{e[intent][lang]}"
    return {"intent": intent, "disease": e["name"][lang], "text": text}
