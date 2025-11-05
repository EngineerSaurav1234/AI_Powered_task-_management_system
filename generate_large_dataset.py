import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()

# Categories and priorities
categories = ["Work", "Personal", "Health", "Learning", "Home", "Career"]
priorities = ["Low", "Medium", "High"]
statuses = ["Pending", "In Progress", "Completed"]
assignees = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]

# Generate 1000 tasks
tasks = []
for i in range(1, 1001):
    task_description = fake.sentence(nb_words=random.randint(5, 15))
    priority = random.choice(priorities)
    status = random.choice(statuses)
    category = random.choice(categories)
    due_date = fake.date_between(start_date='today', end_date='+1y').strftime('%Y-%m-%d')
    assigned_to = random.choice(assignees)
    estimated_hours = random.randint(1, 10)

    tasks.append({
        'task_id': i,
        'task_description': task_description,
        'priority': priority,
        'status': status,
        'category': category,
        'due_date': due_date,
        'assigned_to': assigned_to,
        'estimated_hours': estimated_hours
    })

df = pd.DataFrame(tasks)
df.to_csv('Data/large_tasks.csv', index=False)
print("Large dataset generated and saved to Data/large_tasks.csv")
