# Project Overview
This project is designed to help identify and match lost items on campus with items that have been found.
Using classical ML techniques (TF-IDF + Cosine Similarity), the system compares text descriptions and
returns the closest matches. The project is deployed online using Streamlit for easy access and testing.

# Live Demo (Streamlit Cloud)

🔗 **App Link:** *<https://campuslostfound-awutsjqdq9ufnb2lfodaas.streamlit.app/>*  
Use this link to test:
- Adding a found item  
- Searching a lost item  
- Viewing match scores and sample results  

# Repository Structure
app.py # Streamlit source code
items.csv # Dataset of found items
requirements.txt # Libraries for deployment
README.md # Documentation (this file)

# Dataset Format (items.csv)
The dataset contains sample entries of found items.
**Columns**
- **id** – Auto-generated ID  
- **description** – Item description  
- **location** – Where item was found  
- **date** – Date found  
- **contact** – Finder’s contact  

**Example Rows**
csv
id,description,location,date,contact
1,Black Lenovo laptop bag with red zip,Library stairs,2025-11-20,9876543210
2,Blue steel water bottle with dent,Gym,2025-11-21,9876543211

# Sample Inputs for Testing
Lost Item Search Example
Input:
Black Lenovo laptop bag with red zip
Expected Output:
Match Score: ~90%

# Add Found Item Example

Description:
Grey hoodie jacket found near basketball court
Location:
Basketball Court
Contact:
9876543219

# How Matching Works

Item descriptions are converted into TF-IDF vectors
Similarity between lost & found items is computed using Cosine Similarity
Top matches are displayed with a percentage score

# Sample Images:

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/a5ef3b16-af07-420c-a12f-43efad9cf6fa" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/57f0280f-4072-42e5-90b3-7c7a5fe86025" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/6aecdbc4-0fd3-499f-a87b-347369fb5e0a" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/a30776a5-e1bd-4e6b-91d3-e89d7ea8e90e" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/7785870c-1ac0-42e4-ac93-de19d3d9e5d5" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/956164ab-f777-4dd5-8b84-b4495f71d7bb" />






