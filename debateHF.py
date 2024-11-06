import streamlit as st
from huggingface_hub import InferenceClient
import time
from typing import List, Dict
import json
import os

# Initialize Streamlit page config
st.set_page_config(
    page_title="AI Debate System",
    page_icon="🗣️",
    layout="wide"
)

class HFInferenceLLM:
    """Language model interface using HuggingFace Inference API"""
    def __init__(self, api_token):
        self.client = InferenceClient(
            model="HuggingFaceH4/zephyr-7b-beta",
            token=api_token
        )
    
    def __call__(self, prompt: str) -> str:
        try:
            response = self.client.text_generation(
                prompt,
                max_new_tokens=256,
                temperature=0.7,
                repetition_penalty=1.1
            )
            return response
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "Error generating response"

class DebateAgent:
    """Represents a debater in the system"""
    def __init__(self, name: str, stance: str, llm):
        self.name = name
        self.stance = stance
        self.llm = llm
        self.memory = []
        self.strategy = "balanced"
        self.stats = {
            "arguments_made": 0,
            "rebuttals_made": 0,
            "points_addressed": 0
        }
    
    def remember(self, content: str):
        self.memory.append({
            "content": content,
            "timestamp": time.time()
        })
    
    def generate_opening_statement(self, topic: str, parameters: Dict) -> str:
        """Generates an opening statement"""
        prompt = f"""As someone arguing {self.stance} on the topic: {topic}
        
        Generate a clear, direct opening statement that:
        - Uses {parameters['debate_style']} style
        - Presents {parameters['focus_points']} main points
        - Maintains a {self.strategy} approach
        - Provides specific evidence and examples
        
        Important:
        - Focus on the arguments themselves
        - Don't use debate competition language or address judges/audience
        - Be direct and natural in your communication
        
        Keep your response structured and professional."""

    def generate_rebuttal(self, topic: str, opponent_argument: str, parameters: Dict) -> str:
        """Generates a rebuttal to opponent's argument"""
        prompt = f"""As someone arguing {self.stance} on the topic: {topic}
        
        In response to: {opponent_argument}
        
        Generate a direct rebuttal that:
        - Uses {parameters['debate_style']} style
        - Addresses {parameters['focus_points']} key points
        - Maintains a {self.strategy} approach
        - Provides counterarguments and evidence
        
        Important:
        - Focus on addressing the arguments directly
        - Don't use formal debate language or address judges/audience
        - Keep the tone natural and focused on the discussion
        
        Keep your response clear and professional."""
    
    def generate_closing_statement(self, topic: str, parameters: Dict) -> str:
        """Generates a closing statement"""
        memory_points = "\n".join([m["content"] for m in self.memory])
        
        prompt = f"""As someone concluding arguments {self.stance} the topic: {topic}
        
        Previous points made:
        {memory_points}
        
        Generate a closing statement that:
        - Uses {parameters['debate_style']} style
        - Summarizes your key arguments
        - Reinforces your position with evidence
        - Addresses main counterpoints
        
        Important:
        - Focus on summarizing the arguments clearly
        - Don't use debate competition language
        - Keep the tone direct and natural
        - Avoid addressing judges or audience
        
        Keep your response focused and professional."""        
        response = self.llm(prompt)
        self.remember(response)
        return response

class FactCheckerAgent:
    """Fact checks statements made during debate"""
    def __init__(self, llm):
        self.llm = llm
        self.verified_facts = {}
    
    def check_facts(self, statement: str) -> str:
        """Verifies facts in a statement"""
        prompt = f"""Act as a fact-checker. Analyze this statement:
        
        Statement: {statement}
        
        Provide:
        1. Accuracy Rating (True/Partially True/False)
        2. Key claims identified
        3. Evidence assessment
        4. Important context or caveats
        
        Keep response concise but thorough."""
        
        try:
            result = self.llm(prompt)
            self.verified_facts[statement] = result
            return result
        except Exception as e:
            return f"Fact check error: {str(e)}"

class ModeratorAgent:
    """Manages the debate flow and ensures fair discussion"""
    def __init__(self, llm):
        self.llm = llm
    
    def moderate(self, topic: str, stage: str) -> str:
        """Provides moderation text for current debate stage"""
        prompt = f"""As a debate moderator for the topic: {topic}
        Current stage: {stage}
        
        Provide appropriate moderation text that:
        - Maintains neutrality
        - Guides the discussion
        - Ensures fair participation
        
        Keep your response professional and concise."""
        
        try:
            return self.llm(prompt)
        except Exception as e:
            return f"Moderation error: {str(e)}"

class DebateSystem:
    """Main debate system orchestrating all agents"""
    def __init__(self, topic: str, llm, parameters: Dict):
        self.topic = topic
        self.parameters = parameters
        self.debater_pro = DebateAgent("Proponent", "in favor", llm)
        self.debater_con = DebateAgent("Opponent", "against", llm)
        self.fact_checker = FactCheckerAgent(llm)
        self.moderator = ModeratorAgent(llm)
        self.debate_log = []
    
    def log_event(self, event_type: str, content: str):
        """Records debate events"""
        self.debate_log.append({
            'type': event_type,
            'content': content,
            'timestamp': time.time()
        })
    
    def run_debate_round(self) -> List[Dict]:
        """Runs a complete debate round"""
        # Introduction
        intro = self.moderator.moderate(self.topic, "introduction")
        self.log_event("MODERATOR", intro)
        
        # Opening statements
        for debater in [self.debater_pro, self.debater_con]:
            statement = debater.generate_opening_statement(self.topic, self.parameters)
            self.log_event(f"{debater.name.upper()}", statement)
            
            if self.parameters['fact_checking']:
                fact_check = self.fact_checker.check_facts(statement)
                self.log_event("FACT_CHECK", fact_check)
        
        # Main arguments and rebuttals
        for _ in range(self.parameters['debate_rounds']):
            # Pro rebuttal
            pro_rebuttal = self.debater_pro.generate_rebuttal(
                self.topic,
                self.debate_log[-2]['content'],
                self.parameters
            )
            self.log_event("PROPONENT_REBUTTAL", pro_rebuttal)
            
            if self.parameters['fact_checking']:
                fact_check = self.fact_checker.check_facts(pro_rebuttal)
                self.log_event("FACT_CHECK", fact_check)
            
            # Con rebuttal
            con_rebuttal = self.debater_con.generate_rebuttal(
                self.topic,
                pro_rebuttal,
                self.parameters
            )
            self.log_event("OPPONENT_REBUTTAL", con_rebuttal)
            
            if self.parameters['fact_checking']:
                fact_check = self.fact_checker.check_facts(con_rebuttal)
                self.log_event("FACT_CHECK", fact_check)
        
        # Closing statements
        for debater in [self.debater_pro, self.debater_con]:
            closing = debater.generate_closing_statement(self.topic, self.parameters)
            self.log_event(f"{debater.name.upper()}_CLOSING", closing)
        
        # Moderator closing
        closing = self.moderator.moderate(self.topic, "closing")
        self.log_event("MODERATOR", closing)
        
        return self.debate_log

def main():
    st.title("AI Debate System")
    
    # Sidebar for debate parameters
    st.sidebar.title("Debate Parameters")
    
    parameters = {
        'debate_style': st.sidebar.selectbox(
            "Debate Style:",
            ["Formal", "Casual", "Academic"]
        ),
        'debate_rounds': st.sidebar.slider(
            "Number of Exchange Rounds:",
            min_value=1,
            max_value=5,
            value=2
        ),
        'focus_points': st.sidebar.number_input(
            "Points per Argument:",
            min_value=1,
            max_value=5,
            value=3
        ),
        'fact_checking': st.sidebar.checkbox(
            "Enable Fact Checking",
            value=True
        ),
        'show_thinking': st.sidebar.checkbox(
            "Show Agent Thinking Process",
            value=False
        )
    }
    
    # HuggingFace API setup
    api_token = st.text_input(
        "Enter your HuggingFace API token:",
        type="password",
        help="Get your free token at https://huggingface.co/settings/tokens"
    )
    
    if not api_token:
        st.warning("Please enter your HuggingFace API token to continue")
        st.markdown("""
        To get your free API token:
        1. Go to [HuggingFace](https://huggingface.co/join)
        2. Create an account or sign in
        3. Go to Settings → Access Tokens
        4. Create a new token
        """)
        return
    
    # Topic selection
    topic_options = [
        "Should artificial intelligence be regulated?",
        "Is universal basic income a good idea?",
        "Should social media platforms be responsible for content moderation?",
        "Custom topic"
    ]
    
    topic_selection = st.selectbox("Select debate topic:", topic_options)
    
    if topic_selection == "Custom topic":
        topic = st.text_input("Enter your custom topic:")
    else:
        topic = topic_selection
    
    if st.button("Start Debate"):
        if topic:
            with st.spinner("Initializing debate..."):
                try:
                    llm = HFInferenceLLM(api_token)
                    debate = DebateSystem(topic, llm, parameters)
                    debate_log = debate.run_debate_round()
                    
                    # Display debate
                    for event in debate_log:
                        if event['type'] == "MODERATOR":
                            st.write("🎙️ **Moderator:**")
                            st.markdown(event['content'])
                        elif "REBUTTAL" in event['type']:
                            st.write(f"🔄 **{event['type'].replace('_REBUTTAL', '')} Rebuttal:**")
                            st.markdown(event['content'])
                        elif "CLOSING" in event['type']:
                            st.write(f"🎭 **{event['type'].replace('_CLOSING', '')} Closing:**")
                            st.markdown(event['content'])
                        elif event['type'] in ["PROPONENT", "OPPONENT"]:
                            st.write(f"🗣️ **{event['type']}:**")
                            st.markdown(event['content'])
                        elif event['type'] == "FACT_CHECK":
                            with st.expander("📋 Fact Check"):
                                st.markdown(event['content'])
                        st.markdown("---")
                    
                    # Show debate statistics if enabled
                    if parameters['show_thinking']:
                        st.sidebar.markdown("### Debate Statistics")
                        st.sidebar.json({
                            "Pro Arguments": debate.debater_pro.stats,
                            "Con Arguments": debate.debater_con.stats
                        })
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a debate topic")

if __name__ == "__main__":
    main()
