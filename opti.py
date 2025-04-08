import streamlit as st
import pandas as pd
import random
from datetime import datetime
from openai import OpenAI

# ====== SESSION STATE INITIALIZATION ======
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "product_data" not in st.session_state:
    st.session_state.product_data = None
if "score" not in st.session_state:
    st.session_state.score = None
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None
if "openai_key" not in st.session_state:
    st.session_state.openai_key = None
if "client" not in st.session_state:
    st.session_state.client = None

# ====== CORE FUNCTIONS ======
def initialize_openai_client():
    """Initialize OpenAI client if key is available"""
    if st.session_state.openai_key and st.session_state.openai_key.startswith("sk-"):
        st.session_state.client = OpenAI(api_key=st.session_state.openai_key)
    else:
        st.session_state.client = None

def calculate_score(product_data):
    """Calculates profitability score (1-10)"""
    score = 0
    scoring = {
        'amazon_on_listing': {'No': 4, 'Yes': -4},
        'fba_sellers_count': lambda x: 4 if x <= 3 else -4,
        'buy_box_eligible': {'Yes': 4, 'No': -4},
        'is_profitable': {'Yes': 5, 'No': -5},
        'is_variation': {'No': 4, 'Yes': -4},
        'consistent_trends': {'Yes': 4, 'No': -4},
        'price_inelastic': {'Yes': 4, 'No': -4},
        'sales_rank_trend': {'Decreasing': 4, 'Increasing': -4, 'Stable': 0},
        'estimated_demand': {'High': 5, 'Medium': 0, 'Low': -5},
        'offer_count': lambda x: 4 if x <= 3 else -4
    }
    
    for field, rule in scoring.items():
        if callable(rule): score += rule(product_data[field])
        else: score += rule.get(product_data[field], 0)
    
    return round((score / 38) * 10, 1)

def default_insight(score):
    """Fallback rule-based insights"""
    insights = {
        "high": [
            f"This product shows excellent potential with a score of {score}/10",
            f"Strong profitability indicators detected ({score}/10)",
            f"Market conditions favorable for this product ({score}/10)"
        ],
        "medium": [
            f"Moderate potential with a score of {score}/10 - some optimization needed",
            f"This product shows opportunities with careful planning ({score}/10)",
            f"Mixed indicators - requires strategic approach ({score}/10)"
        ],
        "low": [
            f"Low profitability potential ({score}/10) - consider alternatives",
            f"Significant challenges detected with this product ({score}/10)",
            f"Market conditions unfavorable for this product ({score}/10)"
        ]
    }
    if score >= 7: return random.choice(insights["high"])
    elif score >= 4: return random.choice(insights["medium"])
    else: return random.choice(insights["low"])

def generate_ai_insight(score, product_data):
    """Generates dynamic AI-powered insights"""
    if st.session_state.client is None:
        return default_insight(score)
    
    try:
        prompt = f"""
        As an Amazon FBA expert, analyze this product:
        - Profitability Score: {score}/10
        - Price: ${product_data.get('current_price', 'N/A')}
        - FBA Sellers: {product_data['fba_sellers_count']}
        - Buy Box Eligible: {product_data['buy_box_eligible']}
        - Profit Margin: {product_data['is_profitable']}
        - Market Demand: {product_data['estimated_demand']}

        Provide:
        1. Three key strengths (use ✅ emoji)
        2. Two potential risks (use ⚠️ emoji) 
        3. Three actionable recommendations (use 🚀 emoji)
        Format in markdown with clear headings.
        """
        
        response = st.session_state.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an experienced Amazon marketplace analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"AI Insight Error: {str(e)}")
        return default_insight(score)

def generate_ai_response(user_input, score, product_data, chat_history):
    """Enhanced AI chat responses"""
    if st.session_state.client is None:
        return "Please configure OpenAI API key to enable advanced chat"
    
    try:
        context = "\n".join([msg["content"] for msg in chat_history[-3:]]) if chat_history else ""
        
        prompt = f"""
        Product Analysis Context:
        - Score: {score}/10
        - Data: { {k: v for k, v in product_data.items() if k != 'product_id'} }
        
        Conversation History:
        {context}
        
        User Question: {user_input}
        
        Respond as an Amazon FBA expert with:
        - Data-driven insights
        - Specific recommendations
        - Market context
        - Potential risks
        Use markdown formatting with bullet points.
        """
        
        response = st.session_state.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful Amazon marketplace assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content
        
    except Exception as e:
        return f"⚠️ AI Service Error: {str(e)}"

def generate_score_breakdown(score, product_data):
    """Detailed explanation of scoring factors"""
    return (
        f"### Detailed Score Analysis ({score}/10)\n"
        f"1. **Competition**: {'Low' if product_data['fba_sellers_count'] <= 3 else 'High'} ({product_data['fba_sellers_count']} FBA sellers)\n"
        f"2. **Amazon Competition**: {'Present (-4pts)' if product_data['amazon_on_listing'] == 'Yes' else 'Absent (+4pts)'}\n"
        f"3. **Buy Box**: {'Eligible (+4pts)' if product_data['buy_box_eligible'] == 'Yes' else 'Not eligible (-4pts)'}\n"
        f"4. **Profitability**: {'Positive (+5pts)' if product_data['is_profitable'] == 'Yes' else 'Negative (-5pts)'}\n"
        f"5. **Demand**: {product_data['estimated_demand']} ({'High=+5pts' if product_data['estimated_demand'] == 'High' else 'Low=-5pts'})"
    )

# ====== UI COMPONENTS ======
def show_chat_interface():
    """Displays the interactive chat"""
    st.divider()
    st.subheader("💬 Product Advisor Chat")
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about this analysis..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Generate response with full context
        response = generate_ai_response(
            prompt,
            st.session_state.score,
            st.session_state.product_data,
            st.session_state.chat_history
        )
        
        # Add AI response
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Rerun to update display
        st.rerun()

def show_analysis_form():
    """Displays the input form"""
    with st.form("analysis_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            product_id = st.text_input("ASIN", key="form_asin", placeholder="B08N5KWB9H")
            amazon_on_listing = st.selectbox("Amazon Seller", ["No", "Yes"], key="form_amazon")
            fba_sellers_count = st.slider("FBA Sellers", 0, 50, 2, key="form_fba_sellers")
            buy_box_eligible = st.selectbox("Buy Box Eligible", ["Yes", "No"], key="form_buy_box")
            
        with col2:
            is_profitable = st.selectbox("Profit Margin", ["Yes", "No"], key="form_profitable")
            is_variation = st.selectbox("Variation Listing", ["No", "Yes"], key="form_variation")
            estimated_demand = st.select_slider("Market Demand", ["Low", "Medium", "High"], key="form_demand")
            offer_count = st.slider("Total Sellers", 1, 100, 3, key="form_offers")
        
        st.form_submit_button("Analyze Product", on_click=process_form_submission)

def show_batch_analysis():
    """Handles CSV batch processing"""
    st.subheader("📁 Bulk Analysis via CSV")
    uploaded_file = st.file_uploader("Upload product data", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            required_cols = [
                'amazon_on_listing', 'fba_sellers_count', 'buy_box_eligible',
                'is_profitable', 'is_variation', 'consistent_trends',
                'price_inelastic', 'sales_rank_trend', 'estimated_demand',
                'offer_count'
            ]
            
            if all(col in df.columns for col in required_cols):
                st.success("✅ File validated - ready for analysis")
                
                if st.button("Process Batch Analysis"):
                    results = []
                    for _, row in df.iterrows():
                        try:
                            score = calculate_score(row.to_dict())
                            results.append({
                                **row.to_dict(),
                                'profitability_score': score,
                                'category': "High" if score >=7 else "Moderate" if score >=4 else "Low"
                            })
                        except Exception as e:
                            st.error(f"Error processing row: {e}")
                    
                    st.session_state.batch_results = pd.DataFrame(results)
                    st.rerun()
            
            else:
                st.error("⚠️ Missing required columns in CSV")
                st.code(f"Required columns: {', '.join(required_cols)}")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    if st.session_state.batch_results is not None:
        st.divider()
        st.subheader("Batch Analysis Results")
        st.dataframe(st.session_state.batch_results)
        
        csv = st.session_state.batch_results.to_csv(index=False)
        st.download_button(
            "Download Results",
            data=csv,
            file_name=f"profitability_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

def show_results():
    """Displays analysis results with AI enhancements"""
    st.success("### Analysis Complete")
    
    # Score display
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        st.markdown(f"## Score: {st.session_state.score}/10")
        st.progress(st.session_state.score/10)
    
    # Recommendation
    if st.session_state.score >= 7:
        st.success("✅ Strong Profit Potential")
    elif st.session_state.score >= 4:
        st.warning("🔄 Moderate Potential - Needs Optimization")
    else:
        st.error("❌ Low Profit Potential")
    
    # AI Insight Section
    with st.expander("🔍 AI-Powered Analysis", expanded=True):
        if st.session_state.client is not None:
            with st.spinner("Generating AI insights..."):
                insight = generate_ai_insight(
                    st.session_state.score,
                    st.session_state.product_data
                )
                st.markdown(insight)
        else:
            st.warning("Enable OpenAI API key for advanced insights")
            st.markdown(f"**Basic Insight:** {default_insight(st.session_state.score)}")
    
    # Detailed data
    with st.expander("View Raw Analysis Data"):
        st.json(st.session_state.product_data)
    
    # Chat interface
    show_chat_interface()

# ====== FORM PROCESSING ======
def process_form_submission():
    """Processes form data and calculates score"""
    product_data = {
        'product_id': st.session_state.form_asin,
        'amazon_on_listing': st.session_state.form_amazon,
        'fba_sellers_count': st.session_state.form_fba_sellers,
        'buy_box_eligible': st.session_state.form_buy_box,
        'is_profitable': st.session_state.form_profitable,
        'is_variation': st.session_state.form_variation,
        'estimated_demand': st.session_state.form_demand,
        'offer_count': st.session_state.form_offers,
        'consistent_trends': "Yes",  # Default values
        'price_inelastic': "Yes",
        'sales_rank_trend': "Stable"
    }
    
    st.session_state.product_data = product_data
    st.session_state.score = calculate_score(product_data)
    st.session_state.chat_history = []
    st.session_state.batch_results = None
    st.rerun()

# ====== MAIN APP ======
def main():
    st.set_page_config(page_title="AI Profitability Analyzer", layout="wide")
    st.title("🤖 AI-Powered Amazon Product Analyzer")
    
    # API Key Configuration
    with st.sidebar:
        st.subheader("🔑 OPTISAGE AI Configuration")
        api_key = st.text_input("Enter Secret Key", 
                              value=st.session_state.openai_key if st.session_state.openai_key else "",
                              type="password")
        
        if st.button("Save Key"):
            if api_key.startswith("sk-"):
                st.session_state.openai_key = api_key
                st.session_state.client = OpenAI(api_key=api_key)
                st.success("API key configured!")
                st.rerun()
            else:
                st.error("Invalid API key format (should start with 'sk-')")
        
        if st.session_state.openai_key:
            if st.button("Remove Key"):
                st.session_state.openai_key = None
                st.session_state.client = None
                st.rerun()
    
    # Main app tabs
    tab1, tab2 = st.tabs(["Single Product Analysis", "Bulk CSV Analysis"])
    
    with tab1:
        if st.session_state.product_data is None:
            show_analysis_form()
        else:
            if st.button("← New Analysis"):
                st.session_state.product_data = None
                st.rerun()
            show_results()
    
    with tab2:
        show_batch_analysis()
        if st.session_state.batch_results is not None:
            if st.button("Clear Batch Results"):
                st.session_state.batch_results = None
                st.rerun()

if __name__ == "__main__":
    main()
