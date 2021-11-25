# Detacto

![dedacto.png](images/dedacto.png)

Detacto is a web application rendering any image to the style of your choice. 

![source: [https://www.tensorflow.org/hub/tutorials/tf2_arbitrary_image_stylization](https://www.tensorflow.org/hub/tutorials/tf2_arbitrary_image_stylization)](images/Untitled.png)

source: [https://www.tensorflow.org/hub/tutorials/tf2_arbitrary_image_stylization](https://www.tensorflow.org/hub/tutorials/tf2_arbitrary_image_stylization)

It is based on the model [tf2_arbitrary_image_stylization](https://www.tensorflow.org/hub/tutorials/tf2_arbitrary_image_stylization) hosted on Tensorflow Hub. 

# Getting started

Live demo:
- [https://image-api-6d9.pages.dev](https://image-api-6d9.pages.dev/)

To run and deploy Dedacto, follow the next steps.

# Installation steps

## Requirements
- Docker
- gcloud

## Deploy API

Clone repository to get started
```bash
git clone https://github.com/Youplala/Detacto
cd Detacto/
```

### Setup Firebase

Create a Firebase Storage Project in the [Firebase console](https://console.firebase.google.com). Get your JSON credentials and rename it to `firebase_key.json` in the `API` folder. 

### Deploy application on Google Cloud Run
```bash
gcloud run deploy
```
Then you must follow the instructions such as choosing the app name, the region and the project. You can follow the deployment of the API on [https://console.cloud.google.com/](https://console.cloud.google.com/).

Once the deployment is done, you should copy the url of the deployed application to the clipboard.



## Deploy web application

Once the API deployed, the next step is to deploy the web application. You can use CloudFlare Pages or any other page hosting service. The request will be sent using Javascript and the response will be rendered using the API.

# How it works

Detacto is a web application based on Deep Learning technology. 

The application extracts characteristics from an image in order to apply them on a second image. THese characteristics are made of, for example, object shapes, colors or lights.

Therefore, this technology can find several applications: 

- Entertainment. Diverse image editing applications were created and became very popular on download platforms. Applications such that aging, rejuvination, cartoonization, weight modification, etc, are the most popular. With *Detacto*,  you can have fun transferring styles from famous artists to your own pictures.

- Commerce. This technology allows artists to bring a whole new dimension to their work, adding a new layer of reflexion.

- Social networks. Social network platforms are rising in popularity so much that they are also being used to do marketing campaigns. In order to maintain a coherent style on all these platforms, an artistic direction is needed. *Detacto* is the perfect solution for this. It will allow you to create and maintain a unique style and image for your social networks.

