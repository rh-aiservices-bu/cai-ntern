"""
Generate 100 example traces and log them to MLflow via autolog.

Each prompt is sent as a single-turn chat completion. MLflow autolog
captures every request/response as a trace automatically.

Usage:
    python generate_traces.py
"""

import os
import uuid
import urllib3
import mlflow
from openai import OpenAI

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── MLflow configuration ───────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "https://mlflow.redhat-ods-applications.svc.cluster.local:8443"
EXPERIMENT_NAME     = "it-helpdesk-sdg-finetune"

os.environ["MLFLOW_TRACKING_AUTH"]       = "kubernetes"
os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"

NAMESPACE_PATH = "/run/secrets/kubernetes.io/serviceaccount/namespace"
if os.path.exists(NAMESPACE_PATH):
    with open(NAMESPACE_PATH) as f:
        os.environ["MLFLOW_WORKSPACE"] = f.read().strip()

SA_TOKEN_PATH = "/run/secrets/kubernetes.io/serviceaccount/token"
if os.path.exists(SA_TOKEN_PATH):
    with open(SA_TOKEN_PATH) as f:
        os.environ["MLFLOW_TRACKING_TOKEN"] = f.read().strip()

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"MLflow tracking URI : {MLFLOW_TRACKING_URI}")
print(f"Experiment          : {EXPERIMENT_NAME}")
print(f"Workspace           : {os.environ.get('MLFLOW_WORKSPACE', 'not set')}")

# ── Model configuration ────────────────────────────────────────────────────────
LLM_ENDPOINT = "http://llama-32-predictor.ai501.svc.cluster.local:8080"
MODEL_NAME   = "llama32"

client = OpenAI(
    base_url=LLM_ENDPOINT + "/v1",
    api_key="no-key-required",
)

SYSTEM_PROMPT = (
    "You are an IT Help Desk assistant for a large enterprise. "
    "Help employees resolve technical issues, navigate IT policies, "
    "and get access to the tools and systems they need to do their work."
)

# ── 100 example prompts ────────────────────────────────────────────────────────
PROMPTS = [
    # Password & account access
    "I forgot my Windows password and I'm locked out of my laptop. What do I do?",
    "My Active Directory account got locked after too many failed login attempts. How do I unlock it?",
    "I need to reset my VPN password but the self-service portal isn't loading.",
    "How do I enable multi-factor authentication on my corporate account?",
    "I changed my password but now my email on my phone won't sync. How do I fix it?",
    "My SSO session keeps expiring every hour even though I check 'keep me signed in'. Can you help?",
    "I can't log into Salesforce — it says my account is inactive. Who do I contact?",
    "How do I set up a passkey instead of a password for the employee portal?",
    "My manager left the company and I need access to their shared mailbox. What's the process?",
    "I need a temporary account for a contractor starting Monday. How do I request one?",
    # VPN & remote access
    "The VPN client is showing 'Authentication failed' even though my credentials are correct.",
    "I'm traveling internationally and the VPN is blocked in this country. What are my options?",
    "My VPN connection drops every 30 minutes when I'm on Wi-Fi. How do I fix this?",
    "Which VPN profile should I use when connecting from home vs. a public network?",
    "I can connect to VPN but I can't reach the internal file share at \\\\corp\\\\data. Why?",
    "How do I configure split tunneling on the corporate VPN?",
    "The GlobalProtect agent isn't starting on my Mac after the latest macOS update.",
    "I need VPN access for a vendor who needs to access our staging environment.",
    "Can I use the corporate VPN on my personal laptop, or do I need a company device?",
    "VPN is connected but my Outlook keeps showing 'Disconnected'. What's wrong?",
    # Hardware & devices
    "My laptop won't turn on at all — no lights, no fan. What should I do?",
    "The screen on my work laptop has a crack. How do I get it repaired or replaced?",
    "My keyboard is missing the Fn key function. How do I fix the key mapping?",
    "I spilled coffee on my keyboard. What are the steps I should take right now?",
    "My monitor isn't being detected when I plug it into the docking station.",
    "How do I request a second monitor for my home office setup?",
    "My laptop battery drains in under two hours even when plugged in. Is this covered under warranty?",
    "The USB-C port on my docking station stopped working after a firmware update.",
    "I need a standing desk and ergonomic peripherals. What's the process to request them?",
    "My webcam isn't showing up in Teams or Zoom. The device manager shows a yellow warning.",
    # Software & applications
    "I need to install Python on my work laptop but I don't have admin rights.",
    "How do I request a license for Adobe Creative Cloud?",
    "Microsoft Office keeps crashing when I open large Excel files. How do I troubleshoot this?",
    "I need access to the company's BI tool (Tableau). How do I get a license?",
    "A software update broke my VPN client. How do I roll back to the previous version?",
    "Can I install Slack on my work machine, or is Teams the only approved messaging app?",
    "I'm getting a 'license server unreachable' error when opening AutoCAD.",
    "My Outlook add-ins disappeared after a Windows update. How do I restore them?",
    "I need to run a Linux VM on my Windows laptop for development. Is that allowed?",
    "How do I submit a request for new software that isn't in the approved catalog?",
    # Email & communication
    "I'm not receiving emails from external senders. They say they're sending but nothing arrives.",
    "How do I set up an out-of-office reply in Outlook?",
    "My email signature disappeared after my account was migrated. How do I restore it?",
    "Can I access my corporate email on my personal phone? What do I need to set up?",
    "I accidentally deleted an important email chain. Can it be recovered?",
    "How do I archive old emails to free up my mailbox quota?",
    "I'm getting a lot of spam. How do I report phishing emails correctly?",
    "My calendar isn't syncing between Outlook and my iPhone. How do I fix this?",
    "How do I delegate calendar access to my assistant?",
    "I sent an email to the wrong person with sensitive data. What should I do immediately?",
    # Network & connectivity
    "The office Wi-Fi keeps dropping every few minutes on my laptop but not on my phone.",
    "I can't connect to the corporate Wi-Fi — it says 'Can't connect to this network'.",
    "What's the difference between the 'Corp' and 'Guest' Wi-Fi networks?",
    "My internet is extremely slow since we moved to the new office floor.",
    "I need a wired Ethernet connection at my desk but there's no port nearby.",
    "How do I connect to a network printer on the second floor?",
    "Can I use my personal mobile hotspot for work if the office Wi-Fi is down?",
    "My laptop can't reach any internal websites, but external sites work fine.",
    "I'm getting 'DNS server not responding' errors. What does that mean and how do I fix it?",
    "Is it safe to use the hotel Wi-Fi for work, or do I need the VPN?",
    # Security & compliance
    "I think I clicked a phishing link by mistake. What do I do right now?",
    "I received a suspicious call from someone claiming to be from IT asking for my password.",
    "How do I encrypt a USB drive before putting company data on it?",
    "What types of files am I allowed to store in my personal cloud storage (Google Drive, Dropbox)?",
    "My laptop was stolen at the airport. What are the steps I need to take?",
    "How do I know if my laptop has the required endpoint protection installed?",
    "I need to share a sensitive document with an external partner. What's the approved method?",
    "What's the company policy on using AI tools like ChatGPT for work tasks?",
    "How often am I required to complete the security awareness training?",
    "I found a USB drive in the parking lot. What should I do with it?",
    # File storage & collaboration
    "What's the difference between OneDrive and SharePoint and when should I use each?",
    "I accidentally deleted a file from SharePoint. Can it be restored?",
    "How do I share a large file (over 25 MB) with an external client?",
    "My OneDrive sync is stuck at 'Processing changes' for two days.",
    "How do I request a new shared drive for my team?",
    "I can't edit a SharePoint document — it opens as read-only.",
    "How do I set permissions so only my team can access a specific SharePoint folder?",
    "My local OneDrive folder is using too much disk space. How do I enable Files On-Demand?",
    "Can I access company files from a personal computer when I'm traveling?",
    "The version history on a SharePoint file only shows one version. Why aren't older versions saved?",
    # Onboarding & offboarding
    "I'm a new employee starting today. What accounts and access do I need to set up first?",
    "I just joined the security team. How do I get access to the SIEM dashboard?",
    "A team member is leaving Friday. What's the IT offboarding checklist I need to follow?",
    "How do I transfer all files from a departing employee's OneDrive to their manager?",
    "I'm moving from the finance team to engineering. How do I update my system access?",
    "I need to provision a new laptop for a hire starting remotely next week.",
    "What's the standard software bundle that gets installed on a new employee's laptop?",
    "How long does it take to get access to the code repository after I submit the request?",
    "I need to set up a shared inbox for a new team. What's the process?",
    "A contractor's 90-day access is expiring but the project is extended. How do I renew it?",
    # Incidents & escalations
    "The entire office lost internet 10 minutes ago. Is there a known outage?",
    "My laptop blue-screened and I lost unsaved work. How do I prevent this in the future?",
    "Our team's shared application has been down for an hour. How do I escalate this?",
    "I keep getting a blue screen with error code 0x0000007E. What does this mean?",
    "How do I check the IT status page for known outages before submitting a ticket?",
    "My ticket has been open for three days with no response. How do I escalate it?",
    "Who is the on-call IT contact for critical incidents outside business hours?",
    "The payroll system is down and we have a submission deadline in two hours.",
    "How do I submit a P1 incident ticket for a business-critical system outage?",
    "My team's video call service is down right before a client demo. What are my options?",
]

assert len(PROMPTS) == 100, f"Expected 100 prompts, got {len(PROMPTS)}"


def send_prompt(prompt: str, session_id: str, index: int) -> str:
    with mlflow.start_run(run_name=f"trace-{index:03d}", nested=True):
        mlflow.set_tag("session_id", session_id)
        mlflow.set_tag("prompt_index", index)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content


def main():
    # Enable autolog — captures every OpenAI call as an MLflow trace
    mlflow.openai.autolog()

    session_id = str(uuid.uuid4())
    print(f"\nSession ID : {session_id}")
    print(f"Sending {len(PROMPTS)} prompts...\n")

    with mlflow.start_run(run_name="generate-traces"):
        mlflow.set_tag("session_id", session_id)
        mlflow.log_param("num_prompts", len(PROMPTS))
        mlflow.log_param("model", MODEL_NAME)

        for i, prompt in enumerate(PROMPTS, start=1):
            print(f"[{i:3d}/100] {prompt[:80]}")
            try:
                response = send_prompt(prompt, session_id, i)
                print(f"         -> {response[:100]!r}")
            except Exception as e:
                print(f"         ERROR: {e}")

    print(f"\nDone. All traces logged to experiment '{EXPERIMENT_NAME}'.")


if __name__ == "__main__":
    main()
