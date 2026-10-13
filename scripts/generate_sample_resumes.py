"""Generates 10 realistic sample resumes across PDF, DOCX, and TXT, plus edge cases."""

from pathlib import Path
import docx
import pymupdf

DATA_DIR = Path("data/sample_resumes")
DATA_DIR.mkdir(parents=True, exist_ok=True)

RESUMES = [
    {
        "filename": "resume_01_ml_senior.pdf",
        "format": "pdf",
        "text": """ALEX CHEN
Email: alex.chen@example.com | Phone: 555-0101 | San Francisco, CA

SUMMARY
Senior Machine Learning Engineer with 6 years of industry experience developing end-to-end NLP and Computer Vision systems.

EXPERIENCE
Senior Machine Learning Engineer | TechNova Solutions | Jan 2021 - Present
- Architected and deployed large language model inference pipelines using Python, PyTorch, and Hugging Face Transformers, reducing p95 latency by 35%.
- Implemented semantic search engine using sentence-transformers, FAISS vector search, and spaCy for resume and document screening.
- Fine-tuned BERT and RoBERTa models for multi-class document classification, achieving 94% F1-score on legal text datasets.
- Mentored 4 junior ML engineers and led weekly NLP paper reading group.

Machine Learning Engineer | DataCore Systems | Jun 2018 - Dec 2020
- Built machine learning recommendation engines using Python, scikit-learn, XGBoost, and pandas.
- Designed automated feature engineering pipelines and orchestrated workflows with Apache Airflow and Docker.
- Collaborated with backend team to serve REST API endpoints via FastAPI deployed on AWS ECS.

EDUCATION
Master of Science in Computer Science | Stanford University | 2016 - 2018
Bachelor of Science in Computer Science | UC Berkeley | 2012 - 2016

SKILLS
Programming: Python, SQL, C++, Bash
ML/NLP: PyTorch, TensorFlow, Hugging Face, spaCy, scikit-learn, sentence-transformers, BERT, OpenCV
Data & Tools: Pandas, NumPy, Docker, Git, MLflow, AWS, Linux, Airflow

CERTIFICATIONS
AWS Certified Machine Learning - Specialty (2022)
Deep Learning Specialization - Coursera (2019)
"""
    },
    {
        "filename": "resume_02_data_scientist.docx",
        "format": "docx",
        "text": """SARAH MILLER
Seattle, WA | sarah.miller@example.com

PROFESSIONAL SUMMARY
Data Scientist with 4 years of experience delivering actionable predictive models and statistical analyses.

WORK EXPERIENCE
Data Scientist | CloudAnalytics Inc | Mar 2020 - Present
- Developed customer churn prediction models using Python, scikit-learn, LightGBM, and pandas, saving $1.2M annually.
- Built A/B testing framework and statistical hypothesis testing pipelines for product analytics.
- Deployed real-time scoring microservices using Flask and Docker on AWS.
- Created interactive dashboards in Tableau and Streamlit for executive stakeholders.

Junior Data Analyst | RetailMetric | Jul 2018 - Feb 2020
- Extracted and cleaned customer transaction data using SQL and PostgreSQL.
- Performed exploratory data analysis (EDA) and built regression models using R and Python.

EDUCATION
Bachelor of Science in Statistics | University of Washington | 2014 - 2018

TECHNICAL SKILLS
Languages: Python, R, SQL
Libraries: scikit-learn, pandas, numpy, statsmodels, matplotlib, seaborn
Tools: AWS (S3, EC2), Docker, Git, Tableau, Streamlit, Jupyter
"""
    },
    {
        "filename": "resume_03_fullstack_dev.txt",
        "format": "txt",
        "text": """PRIYA PATEL
Bengaluru, India | priya.patel@example.com

SUMMARY
Full Stack Developer with 5 years of experience building scalable web applications and REST APIs.

EXPERIENCE
Lead Full Stack Engineer | Apex Digital | Aug 2021 - Present
- Spearheaded development of SaaS web platform using React, TypeScript, Next.js, and Node.js.
- Designed backend microservices using Python, FastAPI, PostgreSQL, and Redis.
- Implemented CI/CD pipelines with GitHub Actions and deployed containerized apps using Docker on Kubernetes.

Full Stack Developer | InfoSys Dynamics | Jul 2019 - Jul 2021
- Developed responsive web interfaces using JavaScript, React, Redux, and Tailwind CSS.
- Built secure RESTful APIs using Express.js and MongoDB.
- Optimized database queries, reducing query response times by 40%.

EDUCATION
Bachelor of Technology in Information Technology | NIT Trichy | 2015 - 2019

SKILLS
Frontend: React, TypeScript, JavaScript, HTML5, CSS3, Next.js, Redux
Backend: Node.js, Express, Python, FastAPI, PostgreSQL, MongoDB, Redis
DevOps & Cloud: Docker, Kubernetes, AWS, Git, CI/CD, Linux
"""
    },
    {
        "filename": "resume_04_nlp_researcher.pdf",
        "format": "pdf",
        "text": """DR. MARCUS VANCE
marcus.vance@example.com | Boston, MA

PROFESSIONAL PROFILE
NLP Research Scientist with Ph.D. and 3 years post-doctoral industrial research in generative AI and retrieval-augmented generation.

RESEARCH & EXPERIENCE
Research Scientist | DeepMind Innovations | Sep 2022 - Present
- Researched dense passage retrieval and cross-encoder re-ranking using PyTorch, Transformers, and sentence-transformers.
- Published 2 papers at ACL and EMNLP on hallucination reduction in LLMs.
- Implemented low-rank adaptation (LoRA) and fine-tuning pipelines for large models with DeepSpeed and Hugging Face.

Postdoctoral Researcher | MIT CSAIL | Sep 2020 - Aug 2022
- Conducted NLP research on relation extraction and semantic parsing using spaCy, PyTorch, and BERT.

EDUCATION
Ph.D. in Computer Science (NLP focus) | Carnegie Mellon University | 2016 - 2020
Bachelor of Science in Computer Science | Harvard University | 2012 - 2016

CORE COMPETENCIES
NLP & Deep Learning: PyTorch, Hugging Face, Transformers, sentence-transformers, spaCy, NLTK, LangChain
Programming: Python, C++, CUDA
Tools: Git, Linux, Docker, Slurm, WandB
"""
    },
    {
        "filename": "resume_05_devops_cloud.docx",
        "format": "docx",
        "text": """JORDAN LEE
Austin, TX | jordan.lee@example.com

SUMMARY
DevOps and Cloud Infrastructure Engineer with 7 years of experience automating infrastructure and managing Kubernetes clusters.

WORK HISTORY
Senior DevOps Engineer | CloudScale Inc | Jan 2020 - Present
- Designed and maintained multi-region AWS cloud infrastructure using Terraform and Ansible.
- Managed Kubernetes (EKS) clusters hosting 60+ microservices, improving uptime to 99.99%.
- Built automated CI/CD deployment pipelines using GitLab CI, Docker, and Helm.
- Implemented observability stack with Prometheus, Grafana, and ELK stack.

Systems Engineer | Enterprise Cloud | Jun 2017 - Dec 2019
- Automated Linux server provisioning with Bash and Python scripts.
- Maintained network security, firewalls, and IAM role configurations.

EDUCATION
Bachelor of Science in Computer Engineering | University of Texas at Austin | 2013 - 2017

TECHNICAL SKILLS
Cloud: AWS, Azure, GCP
Infrastructure as Code: Terraform, Ansible, CloudFormation
Containers & Orchestration: Docker, Kubernetes, Helm
CI/CD & Monitoring: Jenkins, GitLab CI, GitHub Actions, Prometheus, Grafana
Scripting: Python, Bash, Shell
"""
    },
    {
        "filename": "resume_06_backend_python.pdf",
        "format": "pdf",
        "text": """ELENA ROSTOVA
New York, NY | elena.rostova@example.com

PROFILE
Backend Software Engineer with 4 years of experience building high-throughput data processing systems and microservices.

PROFESSIONAL EXPERIENCE
Backend Software Engineer | FinTech Systems | Jan 2022 - Present
- Designed scalable financial ledger services using Python, Django, PostgreSQL, and Kafka.
- Built async workers using Celery and Redis to process millions of payment events daily.
- Optimized database indexing and partitioning in PostgreSQL.

Software Developer | NextGen Apps | May 2019 - Dec 2021
- Developed RESTful APIs using Python, Flask, and SQLAlchemy.
- Integrated payment gateways (Stripe, PayPal) and managed OAuth2 authentication.
- Wrote automated unit and integration tests using pytest, maintaining 90%+ code coverage.

EDUCATION
Bachelor of Science in Software Engineering | NYU | 2015 - 2019

SKILLS
Languages: Python, Go, SQL
Frameworks: Django, Flask, FastAPI, Celery, SQLAlchemy
Databases: PostgreSQL, MySQL, Redis, Kafka
Tools: Docker, Git, Linux, Pytest, Postman
"""
    },
    {
        "filename": "resume_07_junior_qa.docx",
        "format": "docx",
        "text": """DAVID KIM
Chicago, IL | david.kim@example.com

SUMMARY
Junior QA & Test Automation Engineer with 1.5 years of experience in manual and automated software testing.

EXPERIENCE
Junior QA Automation Engineer | QualityFirst Labs | Oct 2022 - Present
- Developed automated test scripts using Python, Selenium, and pytest for web applications.
- Executed regression, functional, and API testing using Postman.
- Logged, tracked, and verified bug fixes in Jira.

QA Intern | SoftwareTech | Jan 2022 - Sep 2022
- Created test cases and executed manual test suites for mobile and web apps.

EDUCATION
Associate Degree in Information Technology | City Colleges of Chicago | 2020 - 2022

SKILLS
Testing: Selenium, Pytest, Manual Testing, API Testing, Postman, JIRA
Languages: Python, JavaScript, HTML
"""
    },
    {
        "filename": "resume_08_data_analyst.txt",
        "format": "txt",
        "text": """LUCAS SILVA
Miami, FL | lucas.silva@example.com

PROFESSIONAL SUMMARY
Data Analyst with 3 years of experience in business intelligence, SQL data modeling, and reporting.

WORK EXPERIENCE
Data Analyst | Global Retail Group | Jul 2021 - Present
- Wrote complex SQL queries, stored procedures, and ETL scripts in Snowflake and BigQuery.
- Built automated reporting dashboards in Power BI and Tableau.
- Analyzed marketing funnel conversion rates using Python and pandas.

Junior Business Analyst | Metro Logistics | Jun 2020 - Jun 2021
- Maintained Excel models, financial tracking sheets, and weekly executive KPI reports.

EDUCATION
Bachelor of Business Administration in Information Systems | Florida International University | 2016 - 2020

SKILLS
Data Analysis: SQL, Python, Pandas, Excel (VBA, PowerQuery)
BI Tools: Power BI, Tableau, Looker
Databases: Snowflake, Google BigQuery, PostgreSQL
"""
    },
    {
        "filename": "resume_09_cloud_architect.pdf",
        "format": "pdf",
        "text": """RACHEL ADAMS
Denver, CO | rachel.adams@example.com

EXECUTIVE SUMMARY
Principal Cloud Solutions Architect with 10+ years designing enterprise-grade cloud native platforms and distributed systems.

PROFESSIONAL EXPERIENCE
Principal Cloud Solutions Architect | Summit Cloud Consulting | Feb 2019 - Present
- Directed cloud migrations from on-premise data centers to AWS and Azure for Fortune 500 clients.
- Defined enterprise architecture governance, security policies, and zero-trust networking.
- Led high-availability design for Kubernetes microservices and serverless AWS Lambda workloads.

Lead Infrastructure Engineer | TechScale Enterprise | Jan 2014 - Jan 2019
- Architected automated CI/CD and deployment workflows using Terraform, Jenkins, and Docker.
- Managed Linux server estates and SAN storage arrays.

EDUCATION
Master of Science in Software Engineering | University of Colorado Boulder | 2011 - 2013
Bachelor of Science in Computer Science | Colorado State University | 2007 - 2011

CERTIFICATIONS
AWS Certified Solutions Architect - Professional (2021)
Certified Kubernetes Administrator (CKA) (2020)

CORE COMPETENCIES
Cloud & Architecture: AWS, Azure, GCP, Serverless, Microservices
DevOps & IaC: Terraform, CloudFormation, Ansible, Docker, Kubernetes
Networking & Security: VPC, IAM, OAuth, Zero Trust, TLS
"""
    },
    {
        "filename": "resume_10_frontend_angular.docx",
        "format": "docx",
        "text": """TOM CRANE
Atlanta, GA | tom.crane@example.com

SUMMARY
Frontend Developer with 3 years of experience specializing in Angular, TypeScript, and modern CSS.

WORK EXPERIENCE
Frontend Engineer | Horizon Web Apps | Sep 2021 - Present
- Built enterprise healthcare web portals using Angular 14, TypeScript, RxJS, and NgRx.
- Designed accessible and responsive user interfaces conforming to WCAG 2.1 AA standards.
- Created reusable component libraries with Storybook and SCSS.

Junior Frontend Developer | PixelCraft Studio | Jun 2020 - Aug 2021
- Developed interactive web pages using JavaScript, HTML5, CSS3, and Bootstrap.
- Optimized web asset loading and Core Web Vitals.

EDUCATION
Bachelor of Science in Digital Media | Georgia Tech | 2016 - 2020

SKILLS
Frontend: Angular, TypeScript, JavaScript, RxJS, NgRx, HTML5, CSS3, SCSS, Webpack
Testing: Jasmine, Karma, Cypress
Tools: Git, Figma, JIRA, npm
"""
    }
]

def main():
    print(f"Generating 10 realistic sample resumes into {DATA_DIR}...")
    for item in RESUMES:
        target_path = DATA_DIR / item["filename"]
        fmt = item["format"]
        text = item["text"].strip()

        if fmt == "txt":
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(text)
        elif fmt == "docx":
            doc = docx.Document()
            for line in text.split("\n"):
                if line.isupper() and len(line) < 40 and not ("|" in line or "@" in line):
                    doc.add_heading(line, level=2)
                else:
                    doc.add_paragraph(line)
            doc.save(str(target_path))
        elif fmt == "pdf":
            doc = pymupdf.open()
            page = doc.new_page(width=612, height=792)  # Standard Letter size
            # Insert text line by line to format nicely
            y = 50
            for line in text.split("\n"):
                if y > 750:
                    page = doc.new_page(width=612, height=792)
                    y = 50
                is_hdr = line.isupper() and len(line) < 40 and not ("|" in line or "@" in line)
                font_sz = 13 if is_hdr else 10
                page.insert_text((50, y), line, fontsize=font_sz)
                y += 18 if is_hdr else 14
            doc.save(str(target_path))
        print(f"Generated: {target_path.name} ({fmt.upper()})")

    # Generate edge cases
    # 1. Empty file (0 bytes)
    empty_txt = DATA_DIR / "edge_empty_resume.txt"
    empty_txt.write_bytes(b"")
    print("Generated: edge_empty_resume.txt (Empty 0 bytes)")

    empty_pdf = DATA_DIR / "edge_empty_resume.pdf"
    empty_pdf.write_bytes(b"")
    print("Generated: edge_empty_resume.pdf (Empty 0 bytes)")

    # 2. Corrupt file
    corrupt_pdf = DATA_DIR / "edge_corrupt_resume.pdf"
    corrupt_pdf.write_bytes(b"%PDF-1.4\nCorrupted binary payload\x00\xff\xfe\xfa\x12\x34not a valid pdf")
    print("Generated: edge_corrupt_resume.pdf (Corrupted PDF)")

    corrupt_docx = DATA_DIR / "edge_corrupt_resume.docx"
    corrupt_docx.write_bytes(b"PK\x03\x04corrupted zip payload not a docx")
    print("Generated: edge_corrupt_resume.docx (Corrupted DOCX)")

    # 3. Scanned PDF (single image page with no text layer)
    scanned_pdf = DATA_DIR / "edge_scanned_resume.pdf"
    doc_scanned = pymupdf.open()
    doc_scanned.new_page(width=612, height=792) # Blank page with no text
    doc_scanned.save(str(scanned_pdf))
    print("Generated: edge_scanned_resume.pdf (Scanned/Empty text PDF)")

if __name__ == "__main__":
    main()
