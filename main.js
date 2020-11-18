/* 
Main.js manages the training, classification, and output of sign language gestures. 

- The Main class is responsible for altering page elements on the user interface such as buttons,
video elements, etc. It is also handles the training, prediction, and video call features.
- The PredictionOutput class converts the predicted text passed by Main into text, image, and audio
output. This class is also responsible for turning a caller's words into speech in video call mode.

Credits:
The kNN Classifier used for this project was created by Google TensorFlow. 
The kNN classifier requires the computation of random numbers that is not readily available on JavaScript.
To accomplish this, the work of Johannes Baagøe on "implementations of Randomness in Javascript" was used.
Additionally, usage of TensorFlow was learned from Abishek Singh's "alexa-sign-language-translator".

Author: Sufiyaan Nadeem
*/

// Importing the k-Nearest Neighbors Algorithm
import * as dl from 'deeplearn';

import * as cvstfjs from '@microsoft/customvision-tfjs';
import * as tf from '@tensorflow/tfjs';
import { timeStamp } from 'console';

// Require fs module
const fs = require('fs');

// Webcam Image size. Must be 227.
const IMAGE_SIZE = 224;
// K value for KNN. 10 means that we will take votes from 10 data points to classify each tensor.
const TOPK = 10;
// Percent confidence above which prediction needs to be to return a prediction.
const CONF_THRESHOLD = 0.5

// Initial Gestures that need to be trained.
// The start gesture is for signalling when to start prediction
// The stop gesture is for signalling when to stop prediction
var words = [
  "(idle)", 
  "fine",
  'hello',
  'how',
  'i',
  'no',
  'sorry',
  'thank you',
  'what',
  'yes',
  'you'
];
// var labels = FileReader.readAsText(
//   fs.readFile('./tfjs_model/labels.txt')).split('\n');
// console.log(labels);

/*
The Main class is responsible for the training and prediction of words.
It controls the webcam, user interface, as well as initiates the output of predicted words.
*/
class Main {
  constructor() {
    // Initialize variables for display as well as prediction purposes
    this.exampleCountDisplay = [];
    this.checkMarks = [];
    this.gestureCards = [];
    this.training = -1; // -1 when no class is being trained
    this.videoPlaying = false;
    this.previousPrediction = -1;
    this.currentPredictedWords = [];

    // Variables to restrict prediction rate
    this.now;
    this.then = Date.now();
    this.startTime = this.then;
    this.fps = 5; //framerate - number of prediction per second
    this.fpsInterval = 1000 / this.fps;
    this.elapsed = 0;

    // Initalizing kNN model to none.
    this.knn = null;
    /* Initalizing previous kNN model that we trained when training of the current model
    is stopped or prediction has begun. */
    this.previousKnn = this.knn;

    this.model = this.initializeModel();
    console.log('Model: '+ this.model);

    // words = FileReader.readAsText(fs.readFile('./tfjs_model/labels.txt')).split('\n');
    // console.log(words)

    // Storing all elements that from the User Interface that need to be altered into variables.
    this.welcomeContainer = document.getElementById("welcomeContainer");
    this.proceedBtn = document.getElementById("proceedButton");
    this.proceedBtn.style.display = "block";
    this.proceedBtn.classList.add("animated");
    this.proceedBtn.classList.add("flash");
    this.proceedBtn.addEventListener('click', () => {
      this.welcomeContainer.classList.add("slideOutUp");
    })

    this.stageTitle = document.getElementById("stage");
    this.stageInstruction = document.getElementById("steps");
    this.predButton = document.getElementById("predictButton");
    this.backToTrainButton = document.getElementById("backButton");
    this.nextButton = document.getElementById('nextButton');

    this.statusContainer = document.getElementById("status");
    this.statusText = document.getElementById("status-text");

    this.translationHolder = document.getElementById("translationHolder");
    this.translationText = document.getElementById("translationText");
    this.translatedCard = document.getElementById("translatedCard");

    // this.initialTrainingHolder = document.getElementById('initialTrainingHolder');

    this.videoContainer = document.getElementById("videoHolder");
    this.video = document.getElementById("video");

    // this.trainingContainer = document.getElementById("trainingHolder");
    this.addGestureTitle = document.getElementById("add-gesture");
    this.plusImage = document.getElementById("plus_sign");
    this.addWordForm = document.getElementById("add-word");
    this.newWordInput = document.getElementById("new-word");
    this.doneRetrain = document.getElementById("doneRetrain");
    this.trainingCommands = document.getElementById("trainingCommands");

    // this.videoCallBtn = document.getElementById("videoCallBtn");
    this.videoCall = document.getElementById("videoCall");

    this.trainedCardsHolder = document.getElementById("trainedCardsHolder");

    // Start Translator function is called
    this.initializeTranslator();

    // Instantiate Prediction Output
    this.predictionOutput = new PredictionOutput();
  }

  initializeModel() {
    let model = new cvstfjs.ClassificationModel();
    model.loadModelAsync('tfjs_model/model.json');
    console.log(model);
    console.log(words);
    return model;
  }

  /*This function starts the webcam and initial training process. It also loads the kNN
  classifier*/
  initializeTranslator() {
    this.startWebcam();
    this.createTranslateBtn();
  }

  //This function sets up the webcam
  startWebcam() {
    navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user'
        },
        audio: false
      })
      .then((stream) => {
        this.video.srcObject = stream;
        this.video.width = IMAGE_SIZE;
        this.video.height = IMAGE_SIZE;
        this.video.addEventListener('playing', () => this.videoPlaying = true);
        this.video.addEventListener('paused', () => this.videoPlaying = false);
      })
  }

  /*This function creates the button that goes to the Translate Page. It also initializes the UI 
  of the translate page and starts or stops prediction on click.*/
  createTranslateBtn() {
    this.predButton.style.display = "block";
    // this.createVideoCallBtn(); // create video call button that displays on translate page
    this.createBackToTrainBtn(); // create back to train button that will go back to training page

    this.predButton.addEventListener('click', () => {
      // Change the styling of video display and start prediction
      console.log("go to translate");
      this.video.style.display = "inline-block"; // turn on video from webscam in case it's off

      this.videoCall.style.display = "none"; // turn off video call in case it's on
      // this.videoCallBtn.style.display = "block";

      this.backToTrainButton.style.display = "block";

      // Change style of video display
      this.video.className = "videoPredict";
      this.videoContainer.style.display = "inline-block";
      this.videoContainer.style.width = "";
      this.videoContainer.style.height = "";
      this.videoContainer.className = "videoContainerPredict";
      this.videoContainer.style.border = "8px solid black";


      // Update stage and instruction info
      this.stageTitle.innerText = "Translate";
      this.stageInstruction.innerText = "Start Translating with your Start Gesture.";

      // Remove training UI
      // this.trainingContainer.style.display = "none";
      this.trainedCardsHolder.style.marginTop = "130px";

      // Display translation holder that contains translated text
      this.translationHolder.style.display = "block";

      this.predButton.style.display = "none";
      // Start Translation
      this.setUpTranslation();
      // } else {
      //   alert('You haven\'t added any examples yet.\n\nPress and hold on the "Add Example" button next to each word while performing the sign in front of the webcam.');
      // }
    })
  }

  /*This function stops the training process and allows user's to copy text on the click of
  the translation text.*/
  setUpTranslation() {
    // stop training
    if (this.timer) {
      this.stopTraining();
    }

    // Set status to predict, call copy translated text listener and start prediction
    this.setStatusText("Status: Ready to Predict!", "predict");
    this.video.play();
    this.pred = requestAnimationFrame(this.predict.bind(this));
  }

  /*This function predicts the class of the gesture and returns the predicted text if its above a set threshold.*/
  predict() {
    this.now = Date.now();
    this.elapsed = this.now - this.then;

    if (this.elapsed > this.fpsInterval) {
      this.then = this.now - this.elapsed % this.fpsInterval;
      if (this.videoPlaying) {

        // Extract image from video
        const tfImg = tf.browser.fromPixels(this.video);

        // Resize image
        const smalImg = tf.image.resizeBilinear(tfImg, [IMAGE_SIZE, IMAGE_SIZE]);
        const resized = tf.cast(smalImg, 'float32');

        // Generate 4D Tensor as an Input (Reversed due to the nature of mirrored webcam)
        const t4d = tf.tensor4d(Array.from(resized.dataSync()),[1,IMAGE_SIZE,IMAGE_SIZE,3]).reverse(2);

        // Do prediction
        this.model.executeAsync(t4d).then(result => {
          let arr = result[0];
          let prob = Math.max(...arr);

          if (prob > CONF_THRESHOLD) {
            let label = words[result[0].indexOf(prob)];
            // console.log(label);
  
            this.predictionOutput.textOutput(label);
          }
          

          // Dispose
          tfImg.dispose();
          smalImg.dispose();
          resized.dispose();
          t4d.dispose();
        }).catch(error => {
          tfImg.dispose();
          smalImg.dispose();
          resized.dispose();
          t4d.dispose();
        });
        
        
      }
    }

    // Recursion on predict method
    this.pred = requestAnimationFrame(this.predict.bind(this));
  }

  /*This function pauses the predict method*/
  pausePredicting() {
    console.log("pause predicting");
    this.setStatusText("Status: Paused Predicting", "predict");
    cancelAnimationFrame(this.pred);
    this.previousKnn = this.knn;
  }

  // if predict button is actually a back to training button, stop translation and recreate training UI
  createBackToTrainBtn() {
    this.backToTrainButton.addEventListener('click', () => {
      main.pausePredicting();

      this.stageTitle.innerText = "Continue Training";
      this.stageInstruction.innerText = "Add Gesture Name and Train.";

      this.predButton.innerText = "Translate";
      this.predButton.style.display = "block";
      this.backToTrainButton.style.display = "none";
      this.statusContainer.style.display = "none";

      // Remove all elements from translation mode
      this.video.className = "videoTrain";
      this.videoContainer.className = "videoContainerTrain";
      // this.videoCallBtn.style.display = "none";

      this.translationHolder.style.display = "none";
      this.statusContainer.style.display = "none";

      // Show elements from training mode
      // this.trainingContainer.style.display = "block";
      this.trainedCardsHolder.style.marginTop = "0px";
      this.trainedCardsHolder.style.display = "block";
    });
  }

  /*This function stops the training process*/
  stopTraining() {
    this.video.pause();
    cancelAnimationFrame(this.timer);
    console.log("Knn for start: " + this.knn.getClassExampleCount()[0]);
    this.previousKnn = this.knn; // saves current knn model so it can be used later
  }

  /*This function displays the button that start video call.*/
  // createVideoCallBtn() {
  //   // Display video call feed instead of normal webcam feed when video call btn is clicked
  //   videoCallBtn.addEventListener('click', () => {
  //     this.stageTitle.innerText = "Video Call";
  //     this.stageInstruction.innerText = "Translate Gestures to talk to people on Video Call";

  //     this.video.style.display = "none";
  //     this.videoContainer.style.borderStyle = "none";
  //     this.videoContainer.style.overflow = "hidden";
  //     this.videoContainer.style.width = "630px";
  //     this.videoContainer.style.height = "355px";

  //     this.videoCall.style.display = "block";
  //     this.videoCallBtn.style.display = "none";
  //     this.backToTrainButton.style.display = "none";
  //     this.predButton.innerText = "Local Translation";
  //     this.predButton.style.display = "block";

  //     this.setStatusText("Status: Video Call Activated");
  //   })
  // }

  /*This function sets the status text*/
  setStatusText(status, type) { //make default type thing
    this.statusContainer.style.display = "block";
    this.statusText.innerText = status;
    if (type == "copy") {
      console.log("copy");
      this.statusContainer.style.backgroundColor = "blue";
    } else {
      this.statusContainer.style.backgroundColor = "black";
    }
  }
}

/*
The PredictionOutput class is responsible for turning the translated gesture into text, gesture card, and speech output.
*/
class PredictionOutput {
  constructor() {
    //Initializing variables for speech synthesis and output
    this.synth = window.speechSynthesis;
    this.voices = [];
    this.pitch = 1.0;
    this.rate = 0.9;

    this.statusContainer = document.getElementById("status");
    this.statusText = document.getElementById("status-text");

    this.translationHolder = document.getElementById("translationHolder");
    this.translationText = document.getElementById("translationText");
    this.translatedCard = document.getElementById("translatedCard");
    this.trainedCardsHolder = document.getElementById("trainedCardsHolder");

    this.selectedVoice = 48; // this is Google-US en. Can set voice and language of choice

    this.currentPredictedWords = [];
    this.waitTimeForQuery = 10000;

    this.synth.onvoiceschanged = () => {
      this.populateVoiceList()
    };

    //Set up copy translation event listener
    this.copyTranslation();
  }

  // Checks if speech synthesis is possible and if selected voice is available
  populateVoiceList() {
    if (typeof speechSynthesis === 'undefined') {
      console.log("no synth");
      return;
    }
    this.voices = this.synth.getVoices();

    if (this.voices.indexOf(this.selectedVoice) > 0) {
      console.log(this.voices[this.selectedVoice].name + ':' + this.voices[this.selectedVoice].lang);
    }
  }

  /*This function outputs the word using text and gesture cards*/
  textOutput(word) {

    // If the word is (idle), skip
    if (word == '(idle)') {
      return;
    }

    // If the word is the same as the last word, continue
    if (this.currentPredictedWords[this.currentPredictedWords.length - 1] == word) {
      return;
    }

    // Clean text before start
    if (this.currentPredictedWords.length == 0) {
      this.translationText.innerText = '';
    }

    // Add word to predicted words in this query
    this.currentPredictedWords.push(word);

    // Depending on the word, display the text output
    // if (word == "start") {
    //   this.translationText.innerText += ' ';
    // } else
    this.translationText.innerText += ' ' + word;
    // }

    //Clone Gesture Card
    this.translatedCard.innerHTML = " ";

    // If its not video call mode, speak out the user's word
    if (word != "(idle)") {
      this.speak(word);
    }
  }

  /*This functions clears translation text and cards. Sets the previous predicted words to null*/
  clearPara() {
    this.translationText.innerText = '';
    main.previousPrediction = -1;
    this.currentPredictedWords = []; // empty words in this query
    this.translatedCard.innerHTML = " ";
  }

  /*The function below is adapted from https://stackoverflow.com/questions/45071353/javascript-copy-text-string-on-click/53977796#53977796
  It copies the translated text to the user's clipboard*/
  copyTranslation() {
    this.translationHolder.addEventListener('mousedown', () => {
      main.setStatusText("Text Copied!", "copy");
      const el = document.createElement('textarea'); // Create a <textarea> element
      el.value = this.translationText.innerText; // Set its value to the string that you want copied
      el.setAttribute('readonly', ''); // Make it readonly to be tamper-proof
      el.style.position = 'absolute';
      el.style.left = '-9999px'; // Move outside the screen to make it invisible
      document.body.appendChild(el); // Append the <textarea> element to the HTML document
      const selected =
        document.getSelection().rangeCount > 0 // Check if there is any content selected previously
        ?
        document.getSelection().getRangeAt(0) // Store selection if found
        :
        false; // Mark as false to know no selection existed before
      el.select(); // Select the <textarea> content
      document.execCommand('copy'); // Copy - only works as a result of a user action (e.g. click events)
      document.body.removeChild(el); // Remove the <textarea> element
      if (selected) { // If a selection existed before copying
        document.getSelection().removeAllRanges(); // Unselect everything on the HTML document
        document.getSelection().addRange(selected); // Restore the original selection
      }
    });
  }

  /*This function speaks out the user's gestures. In video call mode, it speaks out the other
  user's words.*/
  speak(word) {
    var utterThis = new SpeechSynthesisUtterance(word);

    utterThis.onerror = function (evt) {
      console.log("Error speaking");
    };

    utterThis.voice = this.voices[this.selectedVoice];
    utterThis.pitch = this.pitch;
    utterThis.rate = this.rate;

    this.synth.speak(utterThis);
  }
}

var main = null;

//Initializes the main class on window load
window.addEventListener('load', () => {
  main = new Main()
});