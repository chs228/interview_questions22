import streamlit as st
import pandas as pd
import uuid
import datetime
import os
from fpdf import FPDF
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Set page config
st.set_page_config(
    page_title="Quiz Generator",
    page_icon="📝",
    layout="centered"
)

# Function to generate enhanced quiz PDF
def generate_quiz_pdf(csv_file, output_pdf, user_name, user_email):
    # Read and sample questions
    df = pd.read_csv(csv_file)
    df = df.sample(n=10)  # Select 10 random questions
    
    # Generate unique quiz code
    quiz_id = str(uuid.uuid4())[:8].upper()
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    class QuizPDF(FPDF):
        def header(self):
            # Logo
            if os.path.exists('logo.png'):
                self.image('logo.png', 10, 8, 33)
            else:
                # Draw a placeholder logo if image doesn't exist
                self.set_fill_color(200, 200, 200)
                self.rect(10, 8, 33, 15, 'F')
                self.set_xy(10, 12)
                self.set_font('Arial', 'B', 10)
                self.set_text_color(80, 80, 80)
                self.cell(33, 8, 'QUIZ LOGO', 0, 0, 'C')
            
            # Header Title
            self.set_font('Arial', 'B', 15)
            self.set_xy(50, 10)
            self.cell(110, 10, 'OFFICIAL QUIZ ASSESSMENT', 0, 0, 'C')
            
            # Date and ID on the right
            self.set_font('Arial', 'I', 8)
            self.set_xy(160, 8)
            self.cell(40, 5, f'Date: {current_date}', 0, 1, 'R')
            self.set_xy(160, 13)
            self.cell(40, 5, f'Quiz ID: {quiz_id}', 0, 1, 'R')
            
            # Header Line
            self.set_draw_color(0, 80, 180)
            self.set_line_width(0.5)
            self.line(10, 25, 200, 25)
            self.ln(20)
        
        def footer(self):
            # Position at 1.5 cm from bottom
            self.set_y(-15)
            
            # Footer line
            self.set_draw_color(0, 80, 180)
            self.set_line_width(0.5)
            self.line(10, self.get_y(), 200, self.get_y())
            
            # Footer text
            self.set_font('Arial', 'I', 8)
            self.set_text_color(128)
            self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
            self.set_x(10)
            self.cell(0, 10, 'Confidential Assessment Document', 0, 0, 'L')
            self.set_x(150)
            self.cell(0, 10, f'Quiz ID: {quiz_id}', 0, 0, 'R')
    
    # Initialize PDF with custom class
    pdf = QuizPDF()
    pdf.alias_nb_pages()  # For total page count in footer
    pdf.set_auto_page_break(auto=True, margin=25)
    pdf.add_page()
    
    # Set watermark
    pdf.set_font('Arial', 'B', 60)
    pdf.set_text_color(230, 230, 230)
    pdf.rotate(45, 105, 150)
    pdf.text(75, 190, 'CONFIDENTIAL')
    pdf.rotate(0)
    
    # User information box
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.rect(10, 35, 190, 40, 'F')
    
    pdf.set_xy(15, 40)
    pdf.cell(50, 8, 'CANDIDATE INFORMATION', 0, 1)
    
    pdf.set_font("Arial", '', 11)
    pdf.set_xy(15, 50)
    pdf.cell(30, 8, 'Name:', 0, 0)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(70, 8, user_name, 0, 1)
    
    pdf.set_font("Arial", '', 11)
    pdf.set_xy(15, 60)
    pdf.cell(30, 8, 'Email:', 0, 0)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(70, 8, user_email, 0, 1)
    
    pdf.set_xy(120, 50)
    pdf.set_font("Arial", '', 11)
    pdf.cell(40, 8, 'Assessment ID:', 0, 0)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(40, 8, quiz_id, 0, 1)
    
    pdf.set_xy(120, 60)
    pdf.set_font("Arial", '', 11)
    pdf.cell(40, 8, 'Date:', 0, 0)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(40, 8, current_date, 0, 1)
    
    # Instructions
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(0, 80, 180)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(190, 10, 'INSTRUCTIONS', 1, 1, 'C', fill=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 10)
    pdf.multi_cell(190, 8, 'Please answer all questions. Each question is worth 10 points. You have 30 minutes to complete this assessment. Write your answers clearly in the space provided.', 1)
    
    # Questions section
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(0, 80, 180)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(190, 10, 'ASSESSMENT QUESTIONS', 1, 1, 'C', fill=True)
    pdf.set_text_color(0, 0, 0)
    
    # Add questions
    for i, (index, row) in enumerate(df.iterrows()):
        question = row['Question']
        
        # Question with background
        pdf.set_font("Arial", 'B', 11)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(190, 10, f"Question {i+1}", 1, 1, 'L', fill=True)
        
        # Question text
        pdf.set_font("Arial", '', 10)
        pdf.multi_cell(190, 8, question, 1)
        
        # Answer space
        pdf.set_font("Arial", 'I', 9)
        pdf.cell(190, 8, 'Answer:', 1, 1, 'L')
        pdf.cell(190, 20, '', 1, 1)  # Empty space for answer
        
        pdf.ln(5)
    
    # Final notes
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 9)
    pdf.cell(190, 8, f'End of assessment. Quiz ID: {quiz_id}', 0, 1, 'C')
    
    # Output the PDF
    pdf.output(output_pdf)
    
    return quiz_id  # Return the quiz ID for reference

# Function to send email with PDF attachment
def send_email(receiver_email, subject, body, pdf_filename, sender_email, sender_password):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    
    # Add body text
    msg.attach(MIMEText(body, 'plain'))
    
    # Add attachment
    with open(pdf_filename, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={pdf_filename}")
        msg.attach(part)
    
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        return True, "Email sent successfully!"
    except Exception as e:
        return False, f"Error sending email: {e}"

# Initialize session state variables if they don't exist
if 'quiz_generated' not in st.session_state:
    st.session_state.quiz_generated = False
if 'email_sent' not in st.session_state:
    st.session_state.email_sent = False
if 'quiz_id' not in st.session_state:
    st.session_state.quiz_id = None

# Main app
st.title("📝 Quiz Generator")

# Check if CSV file exists
csv_file = "quiz_questions.csv"
if not os.path.exists(csv_file):
    st.error(f"Error: The quiz questions file '{csv_file}' was not found in the current directory.")
    st.stop()

# Sidebar for app settings
with st.sidebar:
    st.header("Quiz Settings")
    
    # Display information about the questions file
    try:
        questions_df = pd.read_csv(csv_file)
        st.success(f"✅ Questions loaded: {len(questions_df)} questions available")
        with st.expander("Preview Questions"):
            st.dataframe(questions_df.head(3))
    except Exception as e:
        st.error(f"Error loading questions file: {e}")
    
    # Email settings
    st.subheader("Email Configuration")
    sender_email = st.text_input("Sender Email", "projecttestingsubhash@gmail.com")
    sender_password = st.text_input("App Password", "zgwynxksfnwzusyk", type="password")
    
    # App info
    st.markdown("---")
    st.caption("Quiz Generator v1.0")
    st.caption("© 2025 Quiz Systems")

# Main content
st.markdown("### Request Your Quiz Paper")

# User information form
with st.form("user_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Full Name")
    
    with col2:
        email = st.text_input("Email Address")
    
    # Submit button
    submitted = st.form_submit_button("Generate Quiz")
    
    if submitted:
        if not name or not email:
            st.error("Please fill in all fields.")
        else:
            with st.spinner("Generating your quiz..."):
                # Generate PDF
                output_pdf = f"quiz_{name.replace(' ', '_').lower()}.pdf"
                quiz_id = generate_quiz_pdf(csv_file, output_pdf, name, email)
                
                # Update session state
                st.session_state.quiz_generated = True
                st.session_state.quiz_id = quiz_id
                st.success(f"Quiz generated successfully! Quiz ID: {quiz_id}")

# Show quiz information and email options after generation
if st.session_state.quiz_generated:
    st.markdown("---")
    st.subheader("Quiz Generated!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Quiz ID", st.session_state.quiz_id)
        output_pdf = f"quiz_{name.replace(' ', '_').lower()}.pdf"
        
        # Option to view PDF (using download for now as Streamlit can't display PDFs directly)
        with open(output_pdf, "rb") as pdf_file:
            st.download_button(
                label="Download Quiz PDF",
                data=pdf_file,
                file_name=output_pdf,
                mime="application/pdf"
            )
    
    with col2:
        if not st.session_state.email_sent:
            st.markdown("### Send via Email")
            
            email_subject = st.text_input("Email Subject", "Your Quiz Paper")
            email_body = st.text_area("Email Message", 
                                     f"Hello {name},\n\nAttached is your quiz paper (ID: {st.session_state.quiz_id}).\n\nBest regards,\nQuiz Generator Team")
            
            if st.button("Send Email"):
                with st.spinner("Sending email..."):
                    success, message = send_email(
                        email, 
                        email_subject, 
                        email_body, 
                        output_pdf, 
                        sender_email, 
                        sender_password
                    )
                    
                    if success:
                        st.session_state.email_sent = True
                        st.success(message)
                    else:
                        st.error(message)
        else:
            st.success("Email sent successfully!")
            st.info(f"Quiz paper has been emailed to: {email}")
            
            # Option to send again
            if st.button("Send to Another Email"):
                st.session_state.email_sent = False

# Dashboard section
if st.checkbox("Show Admin Dashboard"):
    st.markdown("---")
    st.subheader("Admin Dashboard")
    
    # Mock data for demonstration
    if 'admin_data' not in st.session_state:
        st.session_state.admin_data = []
    
    # Add current quiz to admin data if available
    if st.session_state.quiz_generated and 'name' in locals() and 'email' in locals():
        new_entry = {
            "name": name, 
            "email": email, 
            "quiz_id": st.session_state.quiz_id, 
            "date": datetime.datetime.now().strftime("%Y-%m-%d"), 
            "sent": st.session_state.email_sent
        }
        
        # Only add if not already in the list
        if not any(entry["quiz_id"] == new_entry["quiz_id"] for entry in st.session_state.admin_data):
            st.session_state.admin_data.append(new_entry)
    
    # Display data in a table
    if st.session_state.admin_data:
        st.dataframe(pd.DataFrame(st.session_state.admin_data))
        
        # Simple metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Quizzes", len(st.session_state.admin_data))
        col2.metric("Emails Sent", sum(1 for item in st.session_state.admin_data if item["sent"]))
        
        sent_percentage = 0
        if len(st.session_state.admin_data) > 0:
            sent_percentage = int(sum(1 for item in st.session_state.admin_data if item["sent"]) / len(st.session_state.admin_data) * 100)
        col3.metric("Completion Rate", f"{sent_percentage}%")
    else:
        st.info("No quiz data available yet. Generate a quiz to see statistics here.")

# Footer
st.markdown("---")
st.caption("This application creates personalized quiz PDFs with unique IDs and can send them via email.")
