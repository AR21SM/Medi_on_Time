# **Medi_on_Time**

**Medi_on_Time** is an AI-powered medication management system designed to improve medication adherence and simplify health routines for the elderly, busy individuals, and anyone struggling with timely intake. By leveraging cutting-edge technologies like OCR, AI/ML models, and personalized reminders, Medi_on_Time ensures better medication safety, adherence tracking, and health outcomes.

---

## **Key Features**
- **Medicine Recognition and Scheduling**  
  Upload photos of medicine bottles/packages. AI uses OCR to recognize text and suggest dosage schedules synced with Google Calendar.

- **Personalized Reminders**  
  Timely push notifications, SMS, or voice calls to remind users to take their medications.

- **Adherence Tracking**  
  Log medication intake, track adherence over time, and generate detailed reports for healthcare providers.

- **Drug Interaction Checker**  
  Alerts for potential drug interactions and food/drink conflicts to ensure safety.

- **Refill Alerts**  
  Monitor medication quantities and receive reminders to refill prescriptions. Optionally connect with local pharmacies.

- **Caregiver Access**  
  Allow caregivers or family members to monitor adherence and receive alerts for missed doses.

- **Health Metrics Logging**  
  Track health metrics like blood pressure or glucose levels and correlate them with medication adherence.

---

## **Tech Stack**
### **Frontend**
- React.js for building a responsive user interface
- TailwindCSS for modern styling
- Google Calendar API integration for scheduling

### **Backend**
- Node.js with Express.js for server-side logic
- MongoDB for secure and scalable data storage
- Integration with ChatGPT or open-source LLMs for interpreting medicine information
- APIs for OCR (e.g., Tesseract.js or Google Cloud Vision)

### **AI & Machine Learning**
- OCR for medicine text recognition
- AI/ML models for personalized insights, drug interaction checks, and adherence predictions

---

## **Folder Structure**
```
medication-assistant/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── common/
│   │   │   ├── layout/
│   │   │   └── features/
│   │   │       ├── MedicineRecognition/
│   │   │       ├── Scheduling/
│   │   │       ├── Reminders/
│   │   │       ├── MedicationTracking/
│   │   │       ├── DrugInteractions/
│   │   │       ├── RefillManagement/
│   │   │       ├── VoiceAssistant/
│   │   │       ├── CaregiverAccess/
│   │   │       ├── HealthMetrics/
│   │   │       ├── MedicationInfo/
│   │   │       └── Telemedicine/
│   ├── pages/
│   ├── services/
│   ├── hooks/
│   ├── utils/
├── backend/
│   ├── src/
│   │   ├── routes/
│   │   ├── controllers/
│   │   ├── models/
│   │   ├── middlewares/
│   │   ├── services/ (OCR, ChatGPT, notifications)
├── ml/ (AI/ML models for OCR, drug interactions)
├── docs/ (Documentation)
├── tests/ (Testing files)
```

---

## **Installation**

### Prerequisites:
Ensure you have the following installed:
- Node.js
- npm or yarn
- MongoDB

### Steps:
1. Clone the repository:
    ```bash
    git clone https://github.com/AR21SM/Medi_on_Time.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Medi_on_Time
    ```
3. Install dependencies:
    - For the frontend:
      ```bash
      cd frontend
      npm install
      ```
    - For the backend:
      ```bash
      cd backend
      npm install
      ```
4. Start the development servers:
    - Frontend:
      ```bash
      npm run dev
      ```
    - Backend:
      ```bash
      npm start
      ```

---

## **Usage**
1. Upload a photo of your medication package to the app.
2. Confirm the suggested schedule generated by AI.
3. Receive reminders via push notifications, SMS, or voice calls.
4. Track your medication adherence and share reports with caregivers or healthcare providers.
5. Get refill alerts and check for potential drug interactions.

---

<div align="center">
  <h1 style="font-size: 36px; color: #555;">Made with ❤️ by AR21SM and Karanveerksb</h1>
</div>
