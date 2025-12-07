/*
variables
*/
var model;
var model_classifier;
var model_regressor_moderate;
var model_regressor_high;
var record_num = 0;

const mu = [0.179756871035941, 1.09654334038055, 0.0166955602536998, 0.0167843551797040];
const sigma = [0.107727066458787, 0.402653600972482, 0.0106738196144654, 0.0127437259595146];


/*
classifier + regressor composite prediction
*/
function predictKcaWithClassifier(input_1, input_2, input_3, input_4, input_5) {
    return tf.tidy(() => {
        // Normalize first 4 features, keep temperature as is
        const xNorm = [
            (input_1 - mu[0]) / sigma[0],
            (input_2 - mu[1]) / sigma[1],
            (input_3 - mu[2]) / sigma[2],
            (input_4 - mu[3]) / sigma[3],
            input_5
        ];
        const xTensor = tf.tensor2d([xNorm], [1, 5]);

        // 1) Classify: moderate (0) vs high (1); classifier has 1 sigmoid output = probHigh
        const classOut = model_classifier.predict(xTensor);  // shape [1, 1]
        const probHigh = classOut.dataSync()[0];
        const theta = probHigh >= 0.5 ? 1 : 0;  // 1 = high-K_ca, 0 = moderate-K_ca

        // 2) Choose corresponding regressor
        const regressor = (theta === 0)
            ? model_regressor_moderate
            : model_regressor_high;

        // 3) Predict K_ca with selected regressor
        const kTensor = regressor.predict(xTensor);
        const Kca = kTensor.dataSync()[0];

        return { Kca, theta, probHigh };
    });
}


/*
display after click
*/
function displayme() {
    // record number
    record_num += 1;
    
    // inputs
    var input_1 = parseFloat(document.getElementById("input_1").value) || 0.0;
    var input_2 = parseFloat(document.getElementById("input_2").value) || 0.0;
    var input_3 = parseFloat(document.getElementById("input_3").value) || 0.0;
    var input_4 = parseFloat(document.getElementById("input_4").value) || 0.0;
    var input_5 = parseFloat(document.getElementById("input_5").value) || 0.0;
    var input_6 = parseFloat(document.getElementById("input_6").value) || 0.0;
    const elem_outk = document.getElementById("output_K");
    const elem_recd = document.getElementById("txtarea");
    
    // evaluation    
    const { Kca: K_o, theta, probHigh } = predictKcaWithClassifier(
        input_1, input_2, input_3, input_4, input_5
    );
    console.log("Classifier probHigh =", probHigh.toFixed(3));
    console.log("Chosen regime θ =", theta === 0 ? "moderate" : "high");
    
    // log: C, Mn, Si, P, S, Cu, Ni, Cr
    var str_output = "C" + input_1.toFixed(3).toString() + 
                     ", Mn" + input_2.toFixed(3).toString() +
                     ", P" + input_3.toFixed(3).toString() +
                     ", S" + input_4.toFixed(3).toString() +
                     ", T" + input_5.toFixed(3).toString() +
                     ", Kca=" + K_o.toFixed(3).toString();
    console.log(str_output);    
                           
    // outputs 
    elem_outk.style.color = "black";
    elem_recd.innerHTML += "Record " + pad(record_num, 3) + 
                           ":  " + str_output + 
                           ";" + "&#13;&#10;";        
        
    elem_outk.value = K_o.toFixed(3);      
}

/*
format record number
*/
function pad(n, width, z) {
  z = z || '0';
  n = n + '';
  return n.length >= width ? n : new Array(width - n.length + 1).join(z) + n;
}

/*
load the model
*/
async function start() { 
    //load the model
    model_classifier = await tf.loadLayersModel('classifier/model.json');
    model_regressor_moderate = await tf.loadLayersModel('regressor_moderate/model.json');
    model_regressor_high = await tf.loadLayersModel('regressor_high/model.json');	
    
    // warm up 
    var a = tf.tensor([[0, 0, 0, 0, 0]]);
    console.log('a shape:', a.shape, a.dtype);
    var pred = model_classifier.predict(a).dataSync();
    console.log('Classifier pred:', pred);
    var pred = model_regressor_moderate.predict(a).dataSync();
    console.log('model_regressor_moderate pred:', pred);	
    var pred = model_regressor_high.predict(a).dataSync();
    console.log('model_regressor_high pred:', pred);    
    //allow running
    allowrun();
}

/*
allow input
*/
function allowrun() {
    document.getElementById('status').innerHTML = 'Model Loaded';
    document.getElementById("run").disabled = false;
}