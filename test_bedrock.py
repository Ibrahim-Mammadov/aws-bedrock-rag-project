import boto3
import json

# Bedrock-u region ilə başladırıq
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="eu-central-1"
)

# Nova modeli üçün sorğu strukturu (inference profile ID istifadə olunur)
payload = {
    "messages": [
        {
            "role": "user",
            "content": [{"text": "Salam! Mənə qısaca de ki, sən kimsən?"}]
        }
    ],
    "inferenceConfig": {
        "max_new_tokens": 300,
        "temperature": 0.7
    }
}

try:
    response = bedrock.invoke_model(
        modelId="eu.amazon.nova-lite-v1:0",
        body=json.dumps(payload)
    )
    
    result = json.loads(response['body'].read())
    answer = result['output']['message']['content'][0]['text']
    
    print("Süni intellektin cavabı:")
    print(answer)

except Exception as e:
    print(f"Xəta baş verdi: {e}")