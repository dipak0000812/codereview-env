"""Procedural Code Review Scenario Generator.

Generates infinite variations of code diffs, dependency graphs, and 
file histories procedurally to ensure robust RL environment quality.
"""

import random
import hashlib
from typing import Dict, List, Optional
from uuid import uuid4

class ProceduralGenerator:
    def __init__(self):
        self.domains = {
            "auth": {
                "files": ["auth.py", "jwt_utils.py", "login.py"],
                "reviewers": ["alice_sec", "bob_auth", "charlie_core"],
                "risk_profile": ["HIGH", "CRITICAL"]
            },
            "ui": {
                "files": ["button.tsx", "layout.css", "header.js"],
                "reviewers": ["dave_frontend", "eve_design"],
                "risk_profile": ["LOW", "MEDIUM"]
            },
            "db": {
                "files": ["models.py", "migrations.sql", "query_builder.py"],
                "reviewers": ["frank_data", "grace_backend"],
                "risk_profile": ["MEDIUM", "HIGH", "CRITICAL"]
            }
        }
    
    def generate(self, task: str, episode_id: Optional[str] = None) -> Dict:
        """Generate a procedural scenario deterministically based on episode_id."""
        seed_str = episode_id if episode_id else str(uuid4())
        seed_int = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed_int)

        # 1. Pick a domain
        domain_name = rng.choice(list(self.domains.keys()))
        domain = self.domains[domain_name]
        
        # 2. Pick a target file
        target_file = rng.choice(domain["files"])
        
        # 3. Determine True Risk & Merge Decision
        true_risk = rng.choice(domain["risk_profile"])
        
        # 4. Generate Diff
        diff = self._generate_diff(rng, domain_name, target_file, true_risk)
        
        # 5. Generate Dependencies (Blast Radius)
        dependency_map, blast_radius = self._generate_dependencies(rng, target_file, list(self.domains.keys()))
        if true_risk == "LOW":
            true_merge = rng.choice(["APPROVE", "APPROVE", "REQUEST_CHANGES"])
        elif true_risk == "CRITICAL":
            true_merge = "BLOCK"
        else:
            true_merge = rng.choice(["APPROVE", "REQUEST_CHANGES", "BLOCK"])
            
        # 6. Generate Reviewers & History
        all_reviewers = [r for d in self.domains.values() for r in d["reviewers"]]
        rng.shuffle(all_reviewers)
        available_reviewers = all_reviewers[:5]
        
        # Make sure an expert is available
        expert = rng.choice(domain["reviewers"])
        if expert not in available_reviewers:
            available_reviewers[0] = expert
            
        file_history = {
            target_file: {
                "commits": rng.randint(5, 50),
                "last_modified_by": expert,
                "incident_rate": round(rng.uniform(0.0, 0.2), 2)
            }
        }

        # For Task 1 and 2, ground truth blast radius / reviewer might not be needed, but we provide it anyway
        return {
            'scenario_id': seed_str,
            'task': task,
            'diff': diff,
            'dependency_map': dependency_map,
            'file_history': file_history,
            'available_reviewers': available_reviewers,
            'ground_truth': {
                'risk_level': true_risk,
                'blast_radius': blast_radius if true_risk != "LOW" else [],
                'recommended_reviewer': expert,
                'merge_decision': true_merge
            }
        }

    def _generate_diff(self, rng: random.Random, domain: str, filename: str, risk: str) -> str:
        """Procedurally generate a code diff."""
        if domain == "auth":
            if risk == "CRITICAL":
                return f"--- a/{filename}\n+++ b/{filename}\n@@ -10,3 +10,3 @@\n def verify_token(token):\n-    if token != secret:\n-        raise Exception()\n+    # TODO: fix token validation\n+    return True"
            elif risk == "HIGH":
                return f"--- a/{filename}\n+++ b/{filename}\n@@ -5,2 +5,2 @@\n-ALGORITHM = 'HS256'\n+ALGORITHM = 'none'"
            else:
                return f"--- a/{filename}\n+++ b/{filename}\n@@ -1,2 +1,2 @@\n-# Auth utils\n+# Authentication utilities"
        elif domain == "ui":
            if risk == "MEDIUM":
                return f"--- a/{filename}\n+++ b/{filename}\n@@ -20,2 +20,3 @@\n function render() {{\n-    return <div>{props.title}</div>\n+    // Possible XSS if title is unescaped HTML\n+    return <div dangerouslySetInnerHTML={{{{__html: props.title}}}} />"
            else:
                return f"--- a/{filename}\n+++ b/{filename}\n@@ -5,2 +5,2 @@\n-color: #333;\n+color: #222;"
        else: # db
            if risk == "CRITICAL":
                return f"--- a/{filename}\n+++ b/{filename}\n@@ -40,3 +40,3 @@\n def get_user(id):\n-    return db.execute('SELECT * FROM users WHERE id = ?', (id,))\n+    return db.execute(f'SELECT * FROM users WHERE id = {{id}}')"
            elif risk == "HIGH":
                return f"--- a/{filename}\n+++ b/{filename}\n@@ -12,2 +12,2 @@\n-class User(Base):\n-    __tablename__ = 'users'\n+class Client(Base):\n+    __tablename__ = 'clients'"
            else:
                return f"--- a/{filename}\n+++ b/{filename}\n@@ -1,2 +1,2 @@\n-# DB models\n+# Database models definition"

    def _generate_dependencies(self, rng: random.Random, target: str, all_domains: List[str]) -> tuple:
        """Generate dependency map and compute blast radius."""
        dep_map = {target: []}
        blast_radius = []
        
        num_deps = rng.randint(0, 4)
        if num_deps > 0:
            for _ in range(num_deps):
                dep = f"service_{rng.randint(1,20)}/api.py"
                dep_map[target].append(dep)
                blast_radius.append(dep)
                
        # Add some noise (other unrelated files)
        num_noise = rng.randint(1, 3)
        for _ in range(num_noise):
            noise_target = f"utils/helper_{rng.randint(1,10)}.py"
            dep_map[noise_target] = [f"app_{rng.randint(1,5)}.py"]
            
        return dep_map, blast_radius

