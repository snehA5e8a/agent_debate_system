import streamlit as st
from huggingface_hub import InferenceClient
import time
from typing import List, Dict
import json
import os

class HFInferenceLLM:
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
    def __init__(self, name: str, stance: str, llm):
        self.name = name
        self.stance = stance
        self.llm = llm
        self.memory = []
        self.strategy = "balanced"  # Can be: aggressive, balanced, analytical
        self.stats = {
            "arguments_made": 0,
            "rebuttals_made": 0,
            "points_addressed": 0
        }
    
    def remember(self, content: str, type: str):
        """Store information in agent's memory"""
        self.memory.append({
            "content": content,
            "type": type,
            "timestamp": time.time()
        })
        
    def analyze_opponent_argument(self, argument: str) -> Dict:
        """Analyze opponent's argument to identify key points"""
        prompt = f"""Analyze this argument and identify:
        1. Main claims
        2. Supporting evidence
        3. Potential weaknesses
        
        Argument: {argument}
        
        Provide analysis in JSON format with these keys."""
        
        analysis = self.llm(prompt)
        st.sidebar.write(f"💭 {self.name}'s Analysis:", analysis)
        return analysis
    
    def generate_opening_statement(self, topic: str, parameters: Dict) -> str:
        """Generate opening statement based on topic and parameters"""
        prompt = f"""As {self.name} arguing {self.stance} on '{topic}',
        generate an opening statement that:
        - Length: {parameters['response_length']}
        - Style: {parameters['debate_style']}
        - Focus on {parameters['focus_points']} main points
        
        Make it {self.strategy} in tone."""
        
        statement = self.llm(prompt)
        self.remember(statement, "opening")
        self.stats["arguments_made"] += 1
        return statement
    
    def generate_rebuttal(self, opponent_argument: str, parameters: Dict) -> str:
        """Generate rebuttal to opponent's argument"""
        analysis = self.analyze_opponent_argument(opponent_argument)
        
        prompt = f"""Based on this analysis: {analysis}
        Generate a rebuttal that:
        - Directly addresses opponent's main points
        - Uses {parameters['debate_style']} style
        - Maintains {self.strategy} approach
        - Supports your stance: {self.stance}"""
        
        rebuttal = self.llm(prompt)
        self.remember(rebuttal, "rebuttal")
        self.stats["rebuttals_made"] += 1
        return rebuttal
    
    def generate_closing_statement(self, parameters: Dict) -> str:
        """Generate closing statement using memory of debate"""
        key_points = [m["content"] for m in self.memory]
        
        prompt = f"""Create a closing statement that:
        - Summarizes your strongest arguments
        - Addresses key counterpoints
        - Reinforces your stance: {self.stance}
        - Uses {parameters['debate_style']} style
        
        Previous points: {key_points}"""
        
        closing = self.llm(prompt)
        self.remember(closing, "closing")
        return closing

class FactCheckerAgent:
    def __init__(self, llm):
        self.llm = llm
        self.verified_facts = {}
        self.fact_check_history = []
    
    def check_facts(self, statement: str, show_process: bool = False) -> Dict:
        """Verify facts in a statement"""
        prompt = """Fact check this statement with these steps:
        1. Identify key claims
        2. Rate each claim (True/Partially True/False)
        3. Provide evidence or context
        4. Note any missing context
        
        Statement: {statement}"""
        
        result = self.llm(prompt)
        
        if show_process:
            st.sidebar.write("🔍 Fact Checking Process:", result)
        
        self.fact_check_history.append({
            "statement": statement,
            "result": result,
            "timestamp": time.time()
        })
        
        return result


class ModeratorAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def moderate(self, topic: str, stage: str) -> str:
        prompt = f"""As a debate moderator discussing {topic}, provide a {stage} statement.
        Be professional, concise, and maintain neutrality.
        Focus on guiding the debate and ensuring fair discussion."""
        
        return self.llm(prompt)


class DebateSystem:
    def __init__(self, topic: str, llm):
        self.topic = topic
        self.debater_pro = DebateAgent("Proponent", "in favor of", llm)
        self.debater_con = DebateAgent("Opponent", "against", llm)
        self.fact_checker = FactCheckerAgent(llm)
        self.moderator = ModeratorAgent(llm)
        self.debate_log = []
        
    def log_event(self, event_type: str, content: str):
        self.debate_log.append({
            'type': event_type,
            'content': content,
            'timestamp': time.time()
        })
        
    def run_debate_round(self) -> List[Dict]:
        # Introduction
        intro = self.moderator.moderate(self.topic, "opening")
        self.log_event("MODERATOR", intro)
        
        # Opening statements
        for debater in [self.debater_pro, self.debater_con]:
            statement = debater.generate_argument(self.topic)
            self.log_event(f"{debater.name.upper()}", statement)
            
            fact_check = self.fact_checker.check_facts(statement)
            self.log_event("FACT_CHECK", fact_check)
        
        # Rebuttals
        for debater in [self.debater_pro, self.debater_con]:
            # Get opponent's last argument
            opponent_arg = self.debate_log[-3]['content'] if debater == self.debater_pro else self.debate_log[-1]['content']
            rebuttal = debater.generate_argument(self.topic, context=opponent_arg)
            self.log_event(f"{debater.name.upper()}_REBUTTAL", rebuttal)
            
            fact_check = self.fact_checker.check_facts(rebuttal)
            self.log_event("FACT_CHECK", fact_check)
        
        # Closing
        closing = self.moderator.moderate(self.topic, "closing")
        self.log_event("MODERATOR", closing)
        
        return self.debate_log
        
def main():
    st.title("AI Debate System")
    
    # Enhanced Initial Parameters
    st.sidebar.title("Debate Parameters")
    
    parameters = {
        "debate_style": st.sidebar.selectbox(
            "Debate Style:",
            ["Formal", "Casual", "Academic"]
        ),
        "response_length": st.sidebar.select_slider(
            "Response Length:",
            options=["Brief", "Moderate", "Detailed"]
        ),
        "focus_points": st.sidebar.number_input(
            "Number of Main Points per Argument:",
            min_value=1, max_value=5, value=3
        ),
        "fact_check_detail": st.sidebar.select_slider(
            "Fact Checking Detail:",
            options=["Basic", "Moderate", "Thorough"]
        ),
        "show_agent_thinking": st.sidebar.checkbox(
            "Show Agent Thinking Process",
            value=True
        )
    }

# Get API token from environment variable or user input
    api_token = os.getenv('HUGGINGFACE_API_TOKEN')
    
    if not api_token:
        api_token = st.text_input(
            "Enter your Hugging Face API token:", 
            type="password",
            help="Get your free token at https://huggingface.co/settings/tokens"
        )
    
    if not api_token:
        st.warning("Please enter your Hugging Face API token to continue")
        st.markdown("""
        To get your free API token:
        1. Go to [Hugging Face](https://huggingface.co/join)
        2. Create an account or sign in
        3. Go to Settings → Access Tokens
        4. Create a new token
        """)
        return
    
    # Initialize LLM
    if 'llm' not in st.session_state:
        try:
            st.session_state['llm'] = HFInferenceLLM(api_token)
            st.success("Successfully connected to Zephyr-7B! 🎉")
        except Exception as e:
            st.error(f"Error connecting to Hugging Face: {str(e)}")
            return
    
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
    
    # Debate format options
    st.sidebar.title("Debate Settings")
    format_options = {
        "Quick": "Single round of arguments",
        "Standard": "Opening statements and rebuttals",
        "Extended": "Multiple rounds with cross-examination"
    }
    debate_format = st.sidebar.selectbox("Select debate format:", list(format_options.keys()))
    
    if st.button("Start Debate"):
        if topic:
            with st.spinner("Generating debate..."):
                try:
                    debate = DebateSystem(topic, st.session_state['llm'])
                    debate_log = debate.run_debate_round()
                    
                    # Display debate with improved formatting
                    for event in debate_log:
                        if event['type'] == "MODERATOR":
                            st.write("🎙️ **Moderator:**")
                            st.markdown(event['content'])
                        elif "REBUTTAL" in event['type']:
                            st.write(f"🔄 **{event['type'].replace('_REBUTTAL', '')} Rebuttal:**")
                            st.markdown(event['content'])
                        elif event['type'] in ["PROPONENT", "OPPONENT"]:
                            st.write(f"🗣️ **{event['type']}:**")
                            st.markdown(event['content'])
                        elif event['type'] == "FACT_CHECK":
                            with st.expander("📋 Fact Check"):
                                st.markdown(event['content'])
                        st.markdown("---")
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a debate topic")

if __name__ == "__main__":
    main()
