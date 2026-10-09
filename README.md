# 🌱 AgriVision AI — Bilingual Voice-Based Plant Health Assistant

AgriVision AI helps farmers and home gardeners identify plant diseases and get easy-to-understand advice. The user shows a photo of an affected leaf, asks a question in **Hindi or English** (by voice or text), and receives the answer as both **text and speech**.

**Live demo:** https://nupursharma4114.github.io/Agrivision-AI/

## Problem

Many farmers and gardeners cannot easily tell what is wrong with a plant, and written advice is often technical, in English, or hard to read. AgriVision AI gives simple guidance in the user's own language, and can speak it aloud.

## How it works

```
📷 See  →  🎤 Ask  →  🧠 Analyze  →  📚 Match knowledge  →  📝 Read + 🔊 Listen  →  🌱 Act
```

1. **Photo:** The user takes a photo with the camera or browses an existing image.
2. **Question:** The user asks by voice or text, e.g. "मेरे टमाटर के पत्तों पर काले धब्बे हैं, क्या करूं?"
3. **Analysis:** The image is analyzed and a likely condition is predicted with a confidence score and alternatives.
4. **Knowledge base:** The condition is matched to curated information: symptoms, causes, treatment, prevention and safety.
5. **Answer:** The app detects what the user is asking (treatment, prevention, causes, safety, symptoms) and replies in the selected language.
6. **Voice:** The reply is shown as text and read aloud using text-to-speech.

## Features

- Photo input: **Take a photo** (camera) or **Browse** (gallery/files)
- Disease prediction with confidence score and other possibilities
- **Result reliability** (High / Medium / Low) combining model confidence and photo quality (brightness, sharpness, leaf size), with tips to retake the photo
- Manual override if the prediction looks wrong
- Hindi and English interface, answers and voice
- Voice questions (speech-to-text) and spoken answers (text-to-speech)
- Follow-up questions in the same conversation
- Knowledge base for 6 conditions: Early Blight, Late Blight, Powdery Mildew, Leaf Rust, nutrient deficiency, healthy leaf
- Safety notes for spraying and handling
- Works in the browser with no server; light and dark themes; mobile friendly

## Tech stack (current prototype)

| Part | Technology |
|---|---|
| App | Single-file HTML, CSS, JavaScript |
| Image analysis | Canvas pixel analysis (colour-pattern classifier) |
| Voice input | Web Speech API (SpeechRecognition) |
| Voice output | Web Speech API (SpeechSynthesis) |
| Camera | MediaDevices `getUserMedia`, with native camera fallback |
| Knowledge base | Bilingual JavaScript data structure |
| Hosting | GitHub Pages |

## Run it

**Online:** open the live demo link above.

**Locally:** download `index.html` and open it in Chrome. No installation is needed.

> Camera and voice features need browser permission and work best in Chrome. Voice recognition may not be available in every browser; you can type questions instead.

## Limitations

- The prototype's disease detection uses simple colour patterns, **not** a trained deep-learning model, so accuracy is limited. Results are guidance only and not a replacement for an agriculture expert.
- The reliability score is an estimate from confidence and photo quality, not a validated accuracy. Real accuracy will be reported (with a confusion matrix) after the MobileNetV2 model is trained and tested.
- Only 6 conditions are covered.
- Treatment advice is general; always follow product labels and local agriculture office recommendations.

## Roadmap

- [ ] Train **MobileNetV2** on the PlantVillage dataset (TensorFlow) and export to **TensorFlow Lite**
- [ ] Build the **Flutter** mobile app with offline detection
- [ ] Store the knowledge base in **SQLite**
- [ ] Add a **FastAPI** backend and RAG over verified agricultural sources for online features
- [ ] Add more crops, diseases and regional languages

## Project structure

```
Agrivision-AI/
├── index.html   # the complete app
└── README.md
```

## AI assistance disclosure

This project was developed by me with AI assistance (Claude by Anthropic) for code generation. The concept, requirements, design decisions and testing are my own.

## Author

**Nupur Sharma** — [GitHub](https://github.com/Nupursharma4114)
