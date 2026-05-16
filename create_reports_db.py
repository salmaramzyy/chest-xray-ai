import os
import json

SAMPLE_REPORTS = [
    {
        "uid": "report_001",
        "findings": "The lungs are clear bilaterally. No focal consolidation, pleural effusion, or pneumothorax is identified. The cardiomediastinal silhouette is within normal limits. Osseous structures are intact.",
        "impression": "No acute cardiopulmonary abnormality.",
        "labels": ["normal"]
    },
    {
        "uid": "report_002",
        "findings": "There is increased opacity in the right lower lobe consistent with consolidation. The left lung is clear. Mild cardiomegaly is present. No pleural effusion identified.",
        "impression": "Right lower lobe pneumonia. Mild cardiomegaly.",
        "labels": ["pneumonia", "cardiomegaly"]
    },
    {
        "uid": "report_003",
        "findings": "Hyperinflation of the lungs bilaterally with flattening of the diaphragms. Increased bronchial markings. No focal consolidation or pleural effusion. Heart size is normal.",
        "impression": "Findings consistent with chronic obstructive pulmonary disease (COPD).",
        "labels": ["emphysema"]
    },
    {
        "uid": "report_004",
        "findings": "Small right-sided pleural effusion. Mild blunting of the right costophrenic angle. The left lung is clear. No pneumothorax. Cardiac silhouette is mildly enlarged.",
        "impression": "Small right pleural effusion. Mild cardiomegaly.",
        "labels": ["pleural_effusion", "cardiomegaly"]
    },
    {
        "uid": "report_005",
        "findings": "No acute infiltrate or consolidation. Lungs are hyperinflated. Mild flattening of the hemidiaphragms bilaterally. No pleural effusion or pneumothorax. Bony thorax is intact.",
        "impression": "Mild pulmonary hyperinflation, possibly related to air trapping or early emphysema.",
        "labels": ["emphysema"]
    },
    {
        "uid": "report_006",
        "findings": "The cardiac silhouette is enlarged with a cardiothoracic ratio greater than 0.5. Mild pulmonary vascular congestion. No frank pulmonary edema. No pleural effusion.",
        "impression": "Cardiomegaly with mild pulmonary vascular congestion, suggestive of early congestive heart failure.",
        "labels": ["cardiomegaly", "heart_failure"]
    },
    {
        "uid": "report_007",
        "findings": "Diffuse bilateral interstitial opacities with a reticular pattern. No focal consolidation. Mild cardiomegaly. No pneumothorax identified.",
        "impression": "Bilateral interstitial opacities, differential includes interstitial lung disease or atypical infection.",
        "labels": ["infiltration"]
    },
    {
        "uid": "report_008",
        "findings": "No focal consolidation or pleural effusion. The lungs are clear. Degenerative changes are noted in the thoracic spine. Heart size is normal.",
        "impression": "No acute pulmonary disease. Degenerative spinal changes.",
        "labels": ["normal"]
    },
    {
        "uid": "report_009",
        "findings": "Left lower lobe atelectasis or consolidation. Small bilateral pleural effusions, left greater than right. Cardiomegaly present. Pulmonary vascular congestion noted.",
        "impression": "Congestive heart failure with bilateral pleural effusions and left lower lobe atelectasis.",
        "labels": ["cardiomegaly", "pleural_effusion", "atelectasis"]
    },
    {
        "uid": "report_010",
        "findings": "There is a 2.5 cm nodular opacity in the right upper lobe. No satellite lesions. Mediastinum is not widened. No pleural effusion. Recommend CT for further evaluation.",
        "impression": "Right upper lobe pulmonary nodule. CT chest recommended for further characterization.",
        "labels": ["nodule"]
    },
    {
        "uid": "report_011",
        "findings": "Patchy bilateral airspace opacities in a perihilar distribution. Bilateral pleural effusions. Cardiomegaly. Findings consistent with pulmonary edema.",
        "impression": "Pulmonary edema with bilateral pleural effusions and cardiomegaly.",
        "labels": ["edema", "cardiomegaly", "pleural_effusion"]
    },
    {
        "uid": "report_012",
        "findings": "Linear atelectasis in the left lower lobe. No focal consolidation or pleural effusion. Heart size and mediastinum are normal. Bony thorax intact.",
        "impression": "Left lower lobe linear atelectasis, likely subsegmental. No acute pneumonia.",
        "labels": ["atelectasis"]
    },
    {
        "uid": "report_013",
        "findings": "Right apical pleural thickening. Old granulomatous disease with calcified hilar lymph nodes. No active infiltrate. Lungs otherwise clear.",
        "impression": "Old granulomatous disease, likely prior tuberculosis exposure. No active disease.",
        "labels": ["pleural_thickening"]
    },
    {
        "uid": "report_014",
        "findings": "The trachea is midline. Bilateral lung fields are clear. No cardiomegaly. No pleural effusion or pneumothorax. Mild thoracic scoliosis noted.",
        "impression": "No acute cardiopulmonary process. Mild thoracic scoliosis.",
        "labels": ["normal"]
    },
    {
        "uid": "report_015",
        "findings": "Large left-sided pleural effusion causing mediastinal shift to the right. Left lung is largely opacified. Right lung is clear. Recommend thoracentesis.",
        "impression": "Large left pleural effusion with mediastinal shift.",
        "labels": ["pleural_effusion"]
    },
]

os.makedirs("data/reports", exist_ok=True)

# Save as JSON
with open("data/reports/knowledge_base.json", "w") as f:
    json.dump(SAMPLE_REPORTS, f, indent=2)

# Also save as individual text files
for r in SAMPLE_REPORTS:
    with open(f"data/reports/{r['uid']}.txt", "w") as f:
        f.write(f"FINDINGS:\n{r['findings']}\n\nIMPRESSION:\n{r['impression']}")

print(f"Created {len(SAMPLE_REPORTS)} radiology reports in data/reports/")
print("Knowledge base saved to data/reports/knowledge_base.json")