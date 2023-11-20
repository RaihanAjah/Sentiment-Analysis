# sentiment-analysis-api
## What is this?
This is a REST API to connect our Tensorflow model into our main api. It is built with Python 3.7.

### Where will this api be applied to project?
When the client post a review for freelance after confirm the order is finished, our main api will take the result from this python api then our main api 
store the result (nlp_score and rating_model_sum) to the database. This rating_model_sum data will be used for sorting/rank the freelancers when the client
search the freelancers from their categories. 

### How to try this API? (currently unavailable)
You can try this api at https://ml-api-4-dot-kerjamin-capstone.et.r.appspot.com/predict 

### Header
```sh
Content-Type: application/json
```
### Body
```sh
{
    "komentar": <komentar-user>,
    "rating": <range:1-5>
}
```
### Response (result)
```sh
{
    "nlp_score": <range:0.0-1.0>,
    "rating_model_sum": <range:0.0-5.0>,
}
```

### Example
![image](https://user-images.githubusercontent.com/83566398/173225421-dc59db10-ca28-4635-b7b8-5f11be163301.png)

