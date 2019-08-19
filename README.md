# Mymnist

An app which can can recognize a user-uploaded image of the handwritten digit.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

First, you need the following tools:

Docker

python3

Download the files, than run the docker build command:
```
docker build --tag=[Name] .
```

To construct the Docker Container:
```
docker run -p 5000:80 [Name]
```

### Submit Prediction

making the correct URL http://localhost:5000.

Go to that URL in a web browser to see the display content served up on a web page.

Click the browser button to upload your picture, then click upload button to get the result.


### Local Testing

Any picture will be changed to the form needed, uplaed your own pictures to see if it get the precise answer.
