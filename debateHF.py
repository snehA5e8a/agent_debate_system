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
                repetition_penalty=1.1,
                return_full_text=False
            )
            if response is None:
                return "No response generated"
            return str(response).strip()
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "Error generating response"

class DebateAgent:
    """
    Represents a debater in the system with agentic properties:
    - Goals: Present compelling arguments for assigned viewpoint
    - Actions: Generate statements, rebuttals, and adapt strategy
    - Perception: Analyze opponent's arguments and debate state
    - Memory: Track argument history and points made
    - Learning: Adapt based on opponent's strategy
    """
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
        
    def remember(self, content: str, type: str):
        """Store information in agent's memory"""
        self.memory.append({
            "content": content,
            "type": type,
            "timestamp": time.time()
        })
        
    def analyze_opponent(self, argument: str) -> Dict:
        """Analyze opponent's argument to adapt strategy"""
        try:
            analysis_prompt = f"""Analyze this argument:
            {argument}
            
            Identify:
            1. Main points
            2. Evidence strength
            3. Emotional vs logical balance
            4. Potential weaknesses"""
            
            analysis = self.llm(analysis_prompt)
            return {"analysis": analysis, "timestamp": time.time()}
        except Exception as e:
            return {"error": str(e)}
    
    def generate_opening_statement(self, topic: str, parameters: Dict) -> str:
        """Generates an opening statement"""
        style_guide = {
            "Formal": "use precise language and academic tone",
            "Casual": "use conversational language while maintaining respect",
            "Academic": "use scholarly language with references to research"
        }
        
        prompt = f"""You are presenting a perspective {self.stance} on: {topic}

Present your opening viewpoint that:
- {style_guide[parameters['debate_style']]}
- Makes {parameters['focus_points']} clear points
- Backs claims with specific examples or evidence
- Maintains a {self.strategy} and constructive tone

Important Guidelines:
- Focus on presenting your perspective directly
- Avoid debate competition language or addressing judges
- Don't use phrases like "honorable judges" or "dear audience"
- Present your points naturally as if in an informed discussion
- Stay focused on the topic and evidence

Begin your response with a clear statement of your position."""
        
        try:
            response = self.llm(prompt)
            if not response or response.isspace():
                response = f"Error: Could not generate opening statement for {self.stance} position"
            self.remember(response, "opening")
            self.stats["arguments_made"] += 1
            return response
        except Exception as e:
            st.error(f"Error in opening statement: {str(e)}")
            return f"Error generating opening statement for {self.stance} position"
    
    def generate_rebuttal(self, topic: str, opponent_argument: str, parameters: Dict) -> str:
        """Generates a rebuttal to opponent's argument"""
        # Analyze opponent's argument
        analysis = self.analyze_opponent(opponent_argument)
        
        style_guide = {
            "Formal": "use precise language and academic tone",
            "Casual": "use conversational language while maintaining respect",
            "Academic": "use scholarly language with references to research"
        }
        
        prompt = f"""You are continuing a discussion {self.stance} on: {topic}

Previous argument to address: {opponent_argument}

Based on analysis: {analysis.get('analysis', 'No analysis available')}

Provide a response that:
- {style_guide[parameters['debate_style']]}
- Addresses {parameters['focus_points']} key points from the previous argument
- Presents counter-evidence or alternative perspectives
- Maintains a {self.strategy} and constructive approach

Important Guidelines:
- Address the arguments directly without debate formalities
- Focus on the substance of the counter-arguments
- Avoid competitive debate language or addressing judges
- Present your points as part of a reasoned discussion
- Keep responses evidence-based and logical

Start by directly addressing the most significant point raised."""
        
        try:
            response = self.llm(prompt)
            if not response or response.isspace():
                response = f"Error: Could not generate rebuttal for {self.stance} position"
            self.remember(response, "rebuttal")
            self.stats["rebuttals_made"] += 1
            return response
        except Exception as e:
            st.error(f"Error in rebuttal: {str(e)}")
            return f"Error generating rebuttal for {self.stance} position"

    def generate_closing_statement(self, topic: str, parameters: Dict) -> str:
        """Generates a closing statement based on debate history"""
        style_guide = {
            "Formal": "use precise language and academic tone",
            "Casual": "use conversational language while maintaining respect",
            "Academic": "use scholarly language with references to research"
        }
        
        # Get previous arguments from memory
        memory_points = "\n".join([f"- {m['content']}" for m in self.memory])
        
        prompt = f"""You are concluding your perspective {self.stance} on: {topic}

Previous points discussed:
{memory_points}

Provide a conclusion that:
- {style_guide[parameters['debate_style']]}
- Synthesizes the main arguments presented
- Reinforces your key evidence and examples
- Addresses significant counterpoints raised

Important Guidelines:
- Summarize your position clearly and directly
- Avoid debate competition language or formalities
- Don't address judges or audience
- Focus on the strength of your arguments and evidence
- Maintain a constructive, solution-oriented tone

Begin with a clear restatement of your main position."""
        
        try:
            response = self.llm(prompt)
            if not response or response.isspace():
                response = f"Error: Could not generate closing statement for {self.stance} position"
            self.remember(response, "closing")
            return response
        except Exception as e:
            st.error(f"Error in closing statement: {str(e)}")
            return f"Error generating closing statement for {self.stance} position"

class FactCheckerAgent:
    """
    Fact checker with agentic properties:
    - Goals: Ensure factual accuracy
    - Actions: Verify claims and provide ratings
    - Perception: Monitor statements
    - Memory: Track verified facts
    - Learning: Improve verification accuracy
    """
    def __init__(self, llm):
        self.llm = llm
        self.verified_facts = {}
        self.verification_history = []
    
    def check_facts(self, statement: str) -> str:
        """Verifies facts in a statement"""
        prompt = f"""As a fact-checker, analyze this statement objectively:

Statement to verify:
{statement}

Provide a structured analysis:
1. Identify specific, verifiable claims
2. Rate each claim (True/Partially True/False) with confidence level
3. Provide relevant context or evidence
4. Note any missing context or potential biases

Format your response clearly with:
- Summary rating
- Individual claim analysis
- Supporting evidence
- Important caveats

Focus on verifiable facts rather than opinions or subjective statements."""
        
        try:
            result = self.llm(prompt)
            self.verified_facts[statement] = result
            self.verification_history.append({
                "statement": statement,
                "result": result,
                "timestamp": time.time()
            })
            return result
        except Exception as e:
            return f"Fact check error: {str(e)}"

class ModeratorAgent:
    """
    Moderator with agentic properties:
    - Goals: Facilitate fair debate
    - Actions: Guide discussion, maintain order
    - Perception: Monitor debate flow
    - Memory: Track debate progress
    - Learning: Adapt moderation style
    """
    def __init__(self, llm):
        self.llm = llm
        self.debate_history = []
    
    def moderate(self, topic: str, stage: str) -> str:
        """Provides moderation text for debate stages"""
        stage_prompts = {
            "introduction": f"""As a moderator, introduce this discussion on: {topic}

Provide a brief, neutral introduction that:
- Clearly states the topic
- Sets expectations for constructive dialogue
- Encourages evidence-based discussion

Keep it concise and focused.""",

            "transition": f"""Guide the discussion on {topic} to the next phase.

Provide a brief transition that:
- Acknowledges points made
- Maintains discussion flow
- Ensures balanced participation

Be concise and neutral.""",

            "closing": f"""Conclude the discussion on: {topic}

Provide a brief closing that:
- Acknowledges key points discussed
- Maintains neutrality
- Emphasizes the value of the discussion

Keep it concise and constructive."""
        }
        
        try:
            response = self.llm(stage_prompts.get(stage, stage_prompts["transition"]))
            self.debate_history.append({
                "stage": stage,
                "content": response,
                "timestamp": time.time()
            })
            return response
        except Exception as e:
            return f"Moderation error: {str(e)}"

class DebateSystem:
    """Main system orchestrating the debate"""
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
        
    def validate_content(self, content: str) -> bool:
        """Check content for inappropriate material"""
        inappropriate_terms = [
            "violent", "abusive", "hate", "discriminatory",
            "threatening", "explicit", "offensive"
        ]
        return not any(term in content.lower() for term in inappropriate_terms)
    
    def run_debate_round(self) -> List[Dict]:
        """Runs a complete debate round"""
        try:
            # Introduction
            intro = self.moderator.moderate(self.topic, "introduction")
            if self.validate_content(intro):
                self.log_event("MODERATOR", intro)
            
            # Opening statements
            for debater in [self.debater_pro, self.debater_con]:
                st.write(f"Generating opening statement for {debater.name}...")
                statement = debater.generate_opening_statement(
                    self.topic, 
                    self.parameters
                )
                
                if self.validate_content(statement):
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
                
                if self.validate_content(pro_rebuttal):
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
                
                if self.validate_content(con_rebuttal):
                    self.log_event("OPPONENT_REBUTTAL", con_rebuttal)
                    
                    if self.parameters['fact_checking']:
                        fact_check = self.fact_checker.check_facts(con_rebuttal)
                        self.log_event("FACT_CHECK", fact_check)
            
            # Closing statements
            for debater in [self.debater_pro, self.debater_con]:
                closing = debater.generate_closing_statement(
                    self.topic,
                    self.parameters
                )
                
                if self.validate_content(closing):
                    self.log_event(f"{debater.name.upper()}_CLOSING", closing)
            
            # Moderator closing
            closing = self.moderator.moderate(self.topic, "closing")
            if self.validate_content(closing):
                self.log_event("MODERATOR", closing)
            
            return self.debate_log
            
        except Exception as e:
            st.error(f"Error in debate round: {str(e)}")
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
        "Is nuclear energy the solution to climate change?",
        "Should remote work become the standard for office jobs?",
        "Custom topic"
    ]
    
    topic_selection = st.selectbox("Select debate topic:", topic_options)
    
    if topic_selection == "Custom topic":
        topic = st.text_input("Enter your custom topic:")
    else:
        topic = topic_selection
    
    # Advanced Settings Expander
    with st.expander("Advanced Debate Settings"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Agent Behavior")
            agent_strategy = st.selectbox(
                "Agent Strategy:",
                ["Balanced", "Analytical", "Persuasive"]
            )
            response_length = st.select_slider(
                "Response Detail Level:",
                options=["Concise", "Moderate", "Detailed"]
            )
        with col2:
            st.markdown("### Content Control")
            content_filters = st.multiselect(
                "Enable Content Filters:",
                ["Fact Verification", "Bias Detection", "Civility Check"],
                default=["Civility Check"]
            )
            real_time_updates = st.checkbox("Show Real-time Agent Thinking", value=False)
    
    if st.button("Start Debate"):
        if topic:
            with st.spinner("Initializing debate..."):
                try:
                    llm = HFInferenceLLM(api_token)
                    
                    # Update parameters with advanced settings
                    parameters.update({
                        'agent_strategy': agent_strategy,
                        'response_length': response_length,
                        'content_filters': content_filters,
                        'real_time_updates': real_time_updates
                    })
                    
                    debate = DebateSystem(topic, llm, parameters)
                    
                    if parameters['show_thinking']:
                        st.sidebar.markdown("### Debate Progress")
                        progress_bar = st.sidebar.progress(0)
                    
                    debate_log = debate.run_debate_round()
                    
                    # Display debate with improved formatting
                    for index, event in enumerate(debate_log):
                        # Update progress if enabled
                        if parameters['show_thinking']:
                            progress = (index + 1) / len(debate_log)
                            progress_bar.progress(progress)
                        
                        # Display event based on type
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
                        
                        # Add separator between events
                        st.markdown("---")
                    
                    # Show debate statistics if enabled
                    if parameters['show_thinking']:
                        st.sidebar.markdown("### Debate Statistics")
                        st.sidebar.json({
                            "Pro Arguments": debate.debater_pro.stats,
                            "Con Arguments": debate.debater_con.stats,
                            "Facts Checked": len(debate.fact_checker.verification_history),
                            "Total Exchanges": parameters['debate_rounds']
                        })
                        
                        # Add download button for debate transcript
                        transcript = "\n\n".join([f"{event['type']}: {event['content']}" 
                                                for event in debate_log])
                        st.download_button(
                            label="Download Debate Transcript",
                            data=transcript,
                            file_name="debate_transcript.txt",
                            mime="text/plain"
                        )
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a debate topic")

if __name__ == "__main__":
    main()
