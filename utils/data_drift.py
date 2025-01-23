import json
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from mlsopsbasic.data.footballDataset import FootballSegmentationDataset
from torchvision import transforms

# Load reference data
reference_dataset = FootballSegmentationDataset(root_dir='data/processed/train')

# Convert dataset to DataFrame
reference_data = []
for idx in range(len(reference_dataset)):
    image, mask = reference_dataset[idx]
    filename = reference_dataset.base_files[idx]
    reference_data.append({"filename": filename, "mask": mask.numpy()})

reference_data = pd.DataFrame(reference_data).head(1)

#dump 1 row of the reference data to a file
reference_data.to_csv('reference_data.csv', index=False)


current_data = pd.read_csv('predictions.csv')
current_data = current_data.rename(columns={"predicted_matrix": "mask"})
#remove the timestamp column
current_data = current_data.drop(columns=["timestamp"])

# Example DataFrame conversion
reference_data['mask'] = reference_data['mask'].apply(lambda x: json.dumps(x.tolist()))
current_data['mask'] = current_data['mask'].apply(lambda x: json.dumps(x))


report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)
report.save_html('report.html')