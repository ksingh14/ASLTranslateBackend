# ASLTranslateBackend
Server for ASL translation from English text to ASL Glosses

## Build 
`docker build -t asl-model-api .`
## Run
`docker run --rm -it -p 8080:8080 asl-model-api`
`curl -X POST -H 'Content-Type: application/json' -d '{"sentence": "John is going to the movies"}' localhost:8080/translate/text`
