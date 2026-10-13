"""Generates 5 realistic sample job descriptions covering various engineering roles."""

from pathlib import Path

JD_DIR = Path("data/sample_jds")
JD_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_JDS = [
    {
        "filename": "jd_01_senior_ml_engineer.txt",
        "text": """Position: Senior Machine Learning Engineer
Department: Artificial Intelligence & NLP
Location: Remote / San Francisco, CA

About the Role:
We are looking for a Senior Machine Learning Engineer to lead the design and deployment of large language models and semantic retrieval systems.

Minimum Qualifications:
- 5+ years of experience building and deploying machine learning models in production environments.
- Master's degree in Computer Science, Data Science, or related quantitative field.
- Strong proficiency in Python, PyTorch, and Hugging Face Transformers.
- Proven track record with Natural Language Processing (NLP), BERT, and Sentence-Transformers.
- Hands-on experience containerizing models using Docker and orchestrating APIs with FastAPI.

Preferred Qualifications:
- Ph.D. in Computer Science with a focus on Deep Learning.
- Experience with FAISS, vector databases, and RAG architectures.
- Familiarity with AWS and Apache Airflow pipelines.
- Knowledge of C++ and CUDA kernel optimization.
"""
    },
    {
        "filename": "jd_02_data_scientist.txt",
        "text": """Job Title: Senior Data Scientist
Company: Retail Analytics Corp

Overview:
We seek an experienced Data Scientist to design predictive models and run experimentation across multi-million user datasets.

Basic Qualifications:
- 4+ years of relevant industry experience in data science and predictive analytics.
- Bachelor of Science in Statistics, Mathematics, or Computer Science.
- Advanced proficiency in Python, SQL, and Pandas.
- Deep expertise in Scikit-learn, LightGBM, and A/B Testing methodologies.
- Experience building dashboards in Tableau.

Desired Skills:
- Experience with AWS cloud infrastructure and Docker.
- Familiarity with R and Streamlit for rapid prototyping.
"""
    },
    {
        "filename": "jd_03_fullstack_engineer.txt",
        "text": """Role: Lead Full Stack Engineer
Location: Bengaluru / Hybrid

Job Summary:
Join our fast-growing SaaS platform building responsive web interfaces and distributed backend APIs.

Requirements:
- Minimum 4 years of professional experience in full stack software engineering.
- Bachelor's degree in Computer Science, Information Technology, or equivalent.
- Strong proficiency in JavaScript, TypeScript, React, and Node.js.
- Strong experience designing REST APIs with FastAPI and Python.
- Proven experience with PostgreSQL, Redis, and Docker.

Nice to Have:
- Experience with Next.js, Redux, and Tailwind CSS.
- Familiarity with Kubernetes and CI/CD pipelines with GitHub Actions.
"""
    },
    {
        "filename": "jd_04_devops_engineer.txt",
        "text": """Job Title: Senior DevOps Engineer
Team: Cloud Platform & Infrastructure

Role Overview:
Help us scale our cloud infrastructure, automate deployment pipelines, and improve system reliability.

Required Experience:
- 5+ years of professional experience managing cloud infrastructure.
- Bachelor's degree in Computer Engineering or related discipline.
- Deep hands-on expertise with AWS, Docker, and Kubernetes.
- Expertise with Infrastructure as Code using Terraform and Ansible.
- Strong shell scripting skills in Bash and Python.
- Proven CI/CD pipeline automation with GitLab CI or Jenkins.

Bonus Points:
- Experience with Helm, Prometheus, and Grafana monitoring stacks.
- Knowledge of GCP and Azure multi-cloud architectures.
"""
    },
    {
        "filename": "jd_05_junior_qa_tester.txt",
        "text": """Job Title: QA Automation Tester
Location: Chicago, IL

Position Overview:
We are looking for an enthusiastic QA Tester to automate our web test suites and maintain test coverage.

Minimum Qualifications:
- At least 1 year of experience in test automation or software QA.
- Associate degree or Diploma in Information Technology or Computer Science.
- Practical experience writing test scripts in Python and Selenium.
- Familiarity with Pytest and API testing with Postman.
- Experience with bug tracking tools like JIRA.

Nice to Have:
- Basic understanding of JavaScript and HTML5.
- Familiarity with CI/CD concepts.
"""
    }
]

def main():
    print(f"Generating 5 sample JDs into {JD_DIR}...")
    for jd in SAMPLE_JDS:
        target = JD_DIR / jd["filename"]
        target.write_text(jd["text"].strip(), encoding="utf-8")
        print(f"Created: {target.name}")

if __name__ == "__main__":
    main()
