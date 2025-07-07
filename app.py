import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from langdetect import detect

# Set page config
st.set_page_config(
    page_title="Ecommerce Intent Classifier",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Translation setup
try:
    from translate import Translator
    translation_enabled = True
except ImportError:
    st.warning("Translation library not available. Using English only mode.")
    translation_enabled = False

# Load models and encoders
@st.cache_resource
def load_models():
    try:
        model = joblib.load('ecommerce_classifier.pkl')
        vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
        label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
        return model, vectorizer, label_encoder
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

model, vectorizer, label_encoder = load_models()

# Define response templates (simplified without translation)
response_templates = {
    "cancel_order": {
        "en": "We've received your request to cancel the order. Our team will process it shortly.",
        "hi": "हमें आपका ऑर्डर रद्द करने का अनुरोध प्राप्त हो गया है। हमारी टीम इसे जल्द ही संसाधित करेगी।",
        "es": "Hemos recibido su solicitud para cancelar el pedido. Nuestro equipo lo procesará en breve."
    },
    "confirm_order": {
        "en": "Your order has been confirmed! You'll receive a confirmation email shortly.",
        "hi": "आपका ऑर्डर पुष्टि हो गया है! आपको जल्द ही एक पुष्टिकरण ईमेल प्राप्त होगा।",
        "es": "¡Su pedido ha sido confirmado! Recibirá un correo electrónico de confirmación en breve."
    },
    "change_address": {
        "en": "To change your delivery address, please provide the new address details.",
        "hi": "अपना डिलीवरी पता बदलने के लिए, कृपया नए पते का विवरण प्रदान करें।",
        "es": "Para cambiar su dirección de entrega, proporcione los detalles de la nueva dirección."
    },
    "contact_advisor": {
        "en": "A customer advisor will contact you shortly. Please provide your preferred contact method.",
        "hi": "एक ग्राहक सलाहकार आपसे जल्द ही संपर्क करेगा। कृपया अपनी पसंदीदा संपर्क विधि प्रदान करें।",
        "es": "Un asesor de clientes se pondrá en contacto con usted en breve. Proporcione su método de contacto preferido."
    },
    "general_query": {
        "en": "Thank you for your query. Our team will respond within 24 hours.",
        "hi": "आपके प्रश्न के लिए धन्यवाद। हमारी टीम 24 घंटों के भीतर जवाब देगी।",
        "es": "Gracias por su consulta. Nuestro equipo responderá en un plazo de 24 horas."
    },
    "get_list_of_products": {
        "en": "Here are some products that might interest you: [Product List]",
        "hi": "यहां कुछ उत्पाद हैं जो आपकी रुचि रखते हैं: [उत्पाद सूची]",
        "es": "Aquí hay algunos productos que podrían interesarle: [Lista de productos]"
    },
    "order_status": {
        "en": "Your order status is: [Status]. Expected delivery date: [Date]",
        "hi": "आपके ऑर्डर की स्थिति है: [स्थिति]। अपेक्षित डिलीवरी तिथि: [तिथि]",
        "es": "El estado de su pedido es: [Estado]. Fecha de entrega prevista: [Fecha]"
    },
    "not_ecommerce": {
        "en": "This doesn't appear to be related to our ecommerce services. Please contact the appropriate department.",
        "hi": "यह हमारी ईकॉमर्स सेवाओं से संबंधित नहीं लगता है। कृपया उचित विभाग से संपर्क करें।",
        "es": "Esto no parece estar relacionado con nuestros servicios de comercio electrónico. Comuníquese con el departamento correspondiente."
    }
}

# Function to detect language
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"  # default to English if detection fails

# Function to translate text (simplified without googletrans)
def translate_to_english(text, src_lang):
    if src_lang == "en" or not translation_enabled:
        return text
    try:
        translator = Translator(to_lang="en", from_lang=src_lang)
        translated = translator.translate(text)
        return translated
    except Exception as e:
        st.warning(f"Translation failed: {e}. Processing original text which may affect accuracy.")
        return text

# Function to classify text
def classify_text(text, language, threshold=0.5):
    try:
        # Translate to English if needed
        if language != "en" and translation_enabled:
            text_to_process = translate_to_english(text, language)
        else:
            text_to_process = text
        
        # Vectorize the text
        vec = vectorizer.transform([text_to_process])
        
        # Predict
        pred = model.predict(vec)[0]
        intent = label_encoder.inverse_transform([pred])[0]
        confidence = model.predict_proba(vec)[0].max()
        
        # Check against threshold
        if confidence < threshold:
            intent = "not_ecommerce"
            confidence = 1.0 - confidence if intent == "not_ecommerce" else confidence
        
        return intent, confidence, text_to_process
    except Exception as e:
        st.error(f"Classification error: {e}")
        return "error", 0.0, text

# Main app function (rest of the code remains the same)
def main():
    st.title("🛒 Ecommerce Intent Classifier")
    st.markdown("Classify customer messages into specific ecommerce intents with multilingual support.")
    
    # Sidebar
    st.sidebar.header("Settings")
    language_options = ["English", "Hindi", "Spanish"]
    if not translation_enabled:
        st.sidebar.warning("Translation disabled - using English only")
        language_options = ["English"]
    
    language = st.sidebar.selectbox("Select Language", language_options, index=0).lower()
    threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05,
                                 help="Adjust the minimum confidence level for ecommerce intent classification")
    
    st.sidebar.markdown("---")
    st.sidebar.header("Example Queries")
    
    # Example queries (simplified)
    example_queries = {
        "en": "I want to cancel my recent order",
        "hi": "मैं अपना हालिया ऑर्डर रद्द करना चाहता हूं",
        "es": "Quiero cancelar mi pedido reciente"
    }
    
    example = st.sidebar.text_area("Try this example:", 
                                 value=example_queries.get(language[:2], example_queries["en"]),
                                 height=50,
                                 key="example_text")
    
    if st.sidebar.button("Use Example"):
        st.session_state.input_text = example
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Supported Intents:**
    - Cancel Order
    - Confirm Order
    - Change Address
    - Contact Advisor
    - General Query
    - Get List of Products
    - Order Status
    - Not Ecommerce
    """)
    
    # Main content
    tab1, tab2 = st.tabs(["Single Text Classification", "Batch Processing"])
    
    with tab1:
        st.subheader("Classify Single Text")
        
        # Text input
        input_text = st.text_area("Enter text to classify:", 
                                value=st.session_state.get("input_text", ""),
                                height=150,
                                key="input_text_area")
        
        if st.button("Classify Text"):
            if input_text.strip():
                # Detect language if auto-detection is enabled
                detected_lang = detect_language(input_text)
                if detected_lang not in ['en', 'hi', 'es']:
                    detected_lang = 'en'  # default to English
                
                st.info(f"Detected language: {detected_lang.upper()} (Processing in {language})")
                
                # Classify
                intent, confidence, translated_text = classify_text(input_text, language[:2], threshold)
                
                # Display results
                st.subheader("Classification Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Input Text:**")
                    st.info(input_text)
                    
                    if translated_text and language != "en" and translation_enabled:
                        st.markdown("**Translated to English:**")
                        st.info(translated_text)
                
                with col2:
                    st.markdown("**Predicted Intent:**")
                    st.success(f"{intent.replace('_', ' ').title()} (Confidence: {confidence:.2%})")
                    
                    st.markdown("**Suggested Response:**")
                    st.info(response_templates[intent].get(language[:2], response_templates[intent]["en"]))
                
                st.markdown("---")
            else:
                st.warning("Please enter some text to classify")
    
    with tab2:
        st.subheader("Batch Process CSV File")
        
        uploaded_file = st.file_uploader("Upload CSV file with 'text' column", type=["csv"])
        
        if uploaded_file is not None:
            if st.button("Process File"):
                with st.spinner("Processing file..."):
                    try:
                        df = pd.read_csv(uploaded_file)
                        if 'text' not in df.columns:
                            st.error("CSV file must contain a 'text' column")
                        else:
                            results = []
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, row in enumerate(df.itertuples()):
                                text = getattr(row, 'text')
                                intent, confidence, _ = classify_text(text, language[:2], threshold)
                                results.append({
                                    'original_text': text,
                                    'predicted_intent': intent,
                                    'confidence': confidence
                                })
                                
                                # Update progress
                                progress = (i + 1) / len(df)
                                progress_bar.progress(progress)
                                status_text.text(f"Processing {i + 1} of {len(df)}...")
                            
                            progress_bar.empty()
                            status_text.empty()
                            
                            results_df = pd.DataFrame(results)
                            st.success("Processing complete!")
                            
                            # Show sample results
                            st.dataframe(results_df.head())
                            
                            # Download button
                            csv = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Results",
                                data=csv,
                                file_name="classification_results.csv",
                                mime="text/csv"
                            )
                            
                            # Visualize results
                            st.subheader("Results Distribution")
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.countplot(data=results_df, y='predicted_intent', ax=ax, 
                                         order=results_df['predicted_intent'].value_counts().index)
                            ax.set_title("Distribution of Predicted Intents")
                            ax.set_xlabel("Count")
                            ax.set_ylabel("Intent")
                            st.pyplot(fig)
                            
                            # Confidence distribution
                            st.subheader("Confidence Distribution")
                            fig2, ax2 = plt.subplots(figsize=(10, 6))
                            sns.histplot(data=results_df, x='confidence', bins=20, kde=True, ax=ax2)
                            ax2.set_title("Confidence Score Distribution")
                            ax2.set_xlabel("Confidence Score")
                            ax2.set_ylabel("Count")
                            st.pyplot(fig2)
                    except Exception as e:
                        st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()