import csv
import random

# Define base intent data with examples per language
base_intents = {
    "cancel_order": {
        "en": [
            "Cancel my order", "I want to cancel this", "Cancel it now", "I made a mistake, cancel", "Abort the purchase",
            "Please cancel my item", "I changed my mind", "Stop the order", "Don't deliver it", "Cancel that product"
        ],
        "hi": [
            "मेरा ऑर्डर रद्द करें", "मैं इसे रद्द करना चाहता हूँ", "इसे अभी रद्द करें", "मुझे ऑर्डर नहीं चाहिए", "ऑर्डर कैंसिल कर दो",
            "कृपया ऑर्डर रद्द करें", "मुझे अब ऑर्डर नहीं चाहिए", "ऑर्डर रोक दो", "डिलीवरी मत करो", "उसे रद्द करो"
        ],
        "es": [
            "Cancela mi pedido", "Quiero cancelar esto", "Cancélalo ahora", "Me equivoqué, cancela", "Anula la compra",
            "Por favor cancela el producto", "Ya no lo quiero", "Detén el pedido", "No lo entregues", "Cancela ese producto"
        ]
    },
    "confirm_order": {
        "en": [
            "Confirm my order", "Yes, place the order", "I want to confirm", "Confirm it now", "Please confirm purchase",
            "I’m okay with this order", "Order confirmed", "I approve this", "Yes, go ahead", "Place it now"
        ],
        "hi": [
            "मेरा ऑर्डर कन्फर्म करें", "हां, ऑर्डर दो", "मैं पुष्टि करना चाहता हूँ", "इसे कन्फर्म करो", "ऑर्डर पक्का करें",
            "मैं सहमत हूँ", "ऑर्डर कन्फर्म है", "इसे स्वीकृत करें", "हां, आगे बढ़ो", "इसे भेजो"
        ],
        "es": [
            "Confirma mi pedido", "Sí, haz el pedido", "Quiero confirmar", "Confírmalo ahora", "Por favor confirma la compra",
            "Estoy de acuerdo con esto", "Pedido confirmado", "Lo apruebo", "Sí, adelante", "Hazlo ya"
        ]
    },
    "order_status": {
        "en": [
            "Where is my order?", "Order status?", "Track my delivery", "Has it shipped?", "Did it get delivered?",
            "Check my order", "Show status", "Is it coming?", "Delivery update please", "What's happening with my order?"
        ],
        "hi": [
            "मेरा ऑर्डर कहाँ है?", "ऑर्डर की स्थिति?", "मेरी डिलीवरी ट्रैक करो", "क्या ऑर्डर भेजा गया?", "क्या डिलीवर हो गया?",
            "ऑर्डर चेक करो", "स्टेटस दिखाओ", "क्या आ रहा है?", "डिलीवरी अपडेट दो", "मेरा ऑर्डर कहाँ पहुंचा?"
        ],
        "es": [
            "¿Dónde está mi pedido?", "¿Estado del pedido?", "Rastrea mi entrega", "¿Fue enviado?", "¿Se entregó ya?",
            "Verifica mi pedido", "Muestra el estado", "¿Está en camino?", "Actualización de entrega", "¿Qué pasa con mi pedido?"
        ]
    },
    "change_address": {
        "en": [
            "Change my address", "I need to update my delivery address", "Edit my address", "New delivery address", "Change the location",
            "I moved, update address", "Change shipping info", "Correct my address", "Update my location", "Modify address"
        ],
        "hi": [
            "मेरा पता बदल दो", "डिलीवरी पता अपडेट करना है", "पता संपादित करें", "नया पता दर्ज करें", "स्थान बदलें",
            "मैं शिफ्ट हो गया हूँ", "शिपिंग जानकारी बदलें", "पता सही करें", "स्थान अपडेट करें", "एड्रेस अपडेट करो"
        ],
        "es": [
            "Cambia mi dirección", "Necesito actualizar la dirección", "Editar dirección", "Nueva dirección de entrega", "Cambiar ubicación",
            "Me mudé, actualiza dirección", "Modificar envío", "Corrige mi dirección", "Actualizar mi ubicación", "Modifica la dirección"
        ]
    },
    "contact_advisor": {
        "en": [
            "I need help", "Contact customer support", "Talk to a human", "Call me", "Connect me to support",
            "Help please", "Customer care", "Can I speak to someone?", "Contact agent", "Support needed"
        ],
        "hi": [
            "मुझे मदद चाहिए", "कस्टमर केयर से संपर्क करें", "किसी से बात कराओ", "मुझे कॉल करो", "सहायता चाहिए",
            "कस्टमर सपोर्ट चाहिए", "एजेंट से बात कराओ", "सहायता करें", "सपोर्ट चाहिए", "मुझे कनेक्ट करो"
        ],
        "es": [
            "Necesito ayuda", "Contactar soporte", "Hablar con alguien", "Llámame", "Conéctame al soporte",
            "Ayuda por favor", "Atención al cliente", "¿Puedo hablar con alguien?", "Contactar agente", "Soporte necesario"
        ]
    },
    "get_list_of_products": {
        "en": [
            "Show me products", "List available items", "What's in stock?", "Product list please", "I want to see products",
            "Give me item list", "Available things", "What can I buy?", "Display all products", "Items available?"
        ],
        "hi": [
            "मुझे उत्पाद दिखाओ", "उपलब्ध आइटम सूची", "स्टॉक में क्या है?", "प्रोडक्ट लिस्ट दो", "मैं आइटम देखना चाहता हूँ",
            "आइटम लिस्ट दो", "क्या-क्या मिल रहा है?", "क्या खरीद सकते हैं?", "सभी उत्पाद दिखाओ", "उपलब्ध आइटम्स?"
        ],
        "es": [
            "Muéstrame los productos", "Lista de artículos disponibles", "¿Qué hay en stock?", "Lista de productos", "Quiero ver productos",
            "Dame lista de artículos", "Cosas disponibles", "¿Qué puedo comprar?", "Mostrar productos", "¿Artículos disponibles?"
        ]
    },
    "not_ecommerce": {
        "en": [
            "Tell me a joke", "What's the weather?", "Sing me a song", "What is your name?", "Who created you?"
        ],
        "hi": [
            "मुझे एक चुटकुला सुनाओ", "मौसम कैसा है?", "एक गाना सुनाओ", "तुम्हारा नाम क्या है?", "तुम्हें किसने बनाया?"
        ],
        "es": [
            "Cuéntame un chiste", "¿Cómo está el clima?", "Cántame una canción", "¿Cuál es tu nombre?", "¿Quién te creó?"
        ]
    }
}

# Expand data to ~1000 rows by random repetition
samples = []
for intent, lang_data in base_intents.items():
    for lang, phrases in lang_data.items():
        # Generate ~50 samples per (intent, lang)
        for _ in range(50):
            sentence = random.choice(phrases)
            samples.append([sentence, intent, lang])

# Save to CSV
output_path = 'data.csv'  # Save in current directory
with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['text', 'intent', 'lang'])
    writer.writerows(samples)

output_path
