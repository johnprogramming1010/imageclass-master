package com.example.loadimage;
// Android imports
import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import android.Manifest;
import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
// Pytorch imports
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
// Java imports
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

    // Add this line at the top of your class to define the tag for Logcat
    private static final String TAG = "MainActivity";
    
    private static final int PERMISSION_REQUEST = 0;    // Request code for permission
    private static final int RESULT_LOAD_IMAGE = 1;    // Request code for loading image

    private final ArrayList<String> subClasses = makeSubClasses();  // List of sub classes
    private final ArrayList<String> superClasses = makeSuperClasses();  // List of super classes
    // Constants
    ImageView imageView;        // Image Box
    Bitmap currImg;             // Used to store the current selected image from the gallery
    Button selectButton;        // Button to select an image
    TextView supClass;          // Super class text view
    Button categorizeButton;    // Button to categorize the image
    TextView subClass;          // Sub class text view

    public MainActivity() {
    }

    // Main function that runs when the app is started
    @RequiresApi(api = Build.VERSION_CODES.TIRAMISU)
    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Log.d(TAG, "onCreate: App started"); // Add this line to indicate that onCreate is called
        imageView = findViewById(R.id.mainImageView);               // Links the image view to the image view in the layout
        selectButton = findViewById(R.id.selectButton);             // Links the select button to the select button in the layout
        categorizeButton = findViewById(R.id.categorizeButton);     // Links the categorize button to the categorize button in the layout
        supClass = findViewById(R.id.superClassCat);                // Links the super class text view to the super class text view in the layout
        subClass = findViewById(R.id.subClassCat);                  // Links the sub class text view to the sub class text view in the layout

        // Request permission to read images from the gallery
        if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, PERMISSION_REQUEST);
        }

        // Selects an image from the gallery and resets the text fields to blank
        selectButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick (View view) {
                Intent intent = new Intent(Intent.ACTION_PICK,
                        MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(intent, RESULT_LOAD_IMAGE);
                subClass.setText("");
                supClass.setText("");
            }

        });

        // Categorizes the image and displays the results in the text fields
        categorizeButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                if (currImg != null) {                                                  // Check if an image has been selected
                    try {
                        float[] fine_results = getFineIndexResults();                       // Get the results from the fine model
                        assert fine_results != null;                                        // Check if the results are not null
                        int maxFineIndex = findMaxIndex(fine_results);                      // Find the index of the maximum value
                        subClass.setText(subClasses.get(maxFineIndex));                     // Display the sub class

                        float[] course_results = getCourseIndexResults();                   // Get the results from the course model
                        assert course_results != null;                                      // Check if the results are not null
                        int maxCourseIndex = findMaxIndex(course_results);                  // Find the index of the maximum value
                        supClass.setText(superClasses.get(maxCourseIndex));                 // Display the super class

                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }


                } else {                                                                // If no image has been selected, display a message
                    Toast.makeText(MainActivity.this, "Please select an image first!", Toast.LENGTH_SHORT).show();
                }
            }
        });
    }

    // Display if permission to read images from the gallery granted or not
    @SuppressLint("MissingSuperCall")
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_REQUEST) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Log.d(TAG, "onRequestPermissionsResult: Permission granted"); // Add this line to indicate permission granted
                Toast.makeText(this, "Permission Granted!", Toast.LENGTH_SHORT).show();
            } else {
                Log.e(TAG, "onRequestPermissionsResult: Permission not granted"); // Add this line to indicate permission not granted
                Toast.makeText(this, "Permission not granted!", Toast.LENGTH_SHORT).show();
                //finish();
            }
        }
    }

    // Load the image from the gallery to the image view
    @SuppressLint("MissingSuperCall")
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == RESULT_LOAD_IMAGE) {
            if (resultCode == RESULT_OK) {
                Uri selectedImage = data.getData();
                if (selectedImage != null) {
                    try {
                        InputStream inputStream = getContentResolver().openInputStream(selectedImage);
                        if (inputStream != null) {
                            Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
                            if (bitmap != null) {
                                imageView.setImageBitmap(bitmap);
                                currImg = bitmap;
                            } else {
                                Log.e(TAG, "onActivityResult: Failed to decode bitmap from input stream.");
                            }
                            inputStream.close();
                        } else {
                            Log.e(TAG, "onActivityResult: InputStream is null.");
                        }
                    } catch (IOException e) {
                        Log.e(TAG, "onActivityResult: Error reading image from URI.", e);
                    }
                } else {
                    Log.e(TAG, "onActivityResult: Selected image URI is null");
                }
            }
        }
    }


    // Load the model and run the image through the model
    private float[] getFineIndexResults() throws IOException {
        if (currImg == null) {          // Check if an image has been selected
            return null;                // Return null if no image has been selected, should not be possible
        }
        Module module = Module.load(assetFilePath("cnn_e16_c100.pt"));                                // Load the model
        Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(currImg,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);   // Convert the image to a tensor
        Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();                        // Run the image through the model
        return  outputTensor.getDataAsFloatArray();                                                       // Return the results
    }

    private float[] getCourseIndexResults() throws IOException {
        if (currImg == null) {          // Check if an image has been selected
            return null;                // Return null if no image has been selected, should not be possible
        }
        Module module = Module.load(assetFilePath("cnn_e15_c20.pt"));                                // Load the model
        Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(currImg,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);   // Convert the image to a tensor
        Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();                        // Run the image through the model
        return  outputTensor.getDataAsFloatArray();                                                       // Return the results
    }

    // Given the name of the pytorch model, get the path for that model
    public String assetFilePath(String assetName) throws IOException {
        File file = new File(this.getFilesDir(), assetName);                // Create a file object
        if (file.exists() && file.length() > 0) {                           // Check if the file exists and has a length greater than 0
            return file.getAbsolutePath();                                  // Return the absolute path of the file
        }

        try (InputStream is = this.getAssets().open(assetName)) {           // Open an input stream to the asset
            try (OutputStream os = Files.newOutputStream(file.toPath())) {  // Open an output stream to the file
                byte[] buffer = new byte[4 * 1024];                         // Create a buffer
                int read;                                                   // Variable to store the number of bytes read                      
                while ((read = is.read(buffer)) != -1) {                    // Read the bytes from the input stream
                    os.write(buffer, 0, read);                          // Write the bytes to the output stream
                }
                os.flush();                                                 // Flush the output stream
            }
            return file.getAbsolutePath();                                  // Return the absolute path of the file
        }
    }

    // Find the index of the maximum value in an array
    private int findMaxIndex(float[] arr) {
        float max = arr[0];                     // Initialize the maximum value to the first element of the array
        int maxIndex = 0;                       // Initialize the index of the maximum value to 0

        for (int i = 1; i < arr.length; i++) {  // Iterate through the array
            if (arr[i] > max) {                 // Check if the current element is greater than the maximum value
                max = arr[i];                       // Update the maximum value
                maxIndex = i;                       // Update the index of the maximum value
            }
        }

        return maxIndex;                        // Return the index of the maximum value
    }

    // Create a list of sub classes
    protected ArrayList<String> makeSubClasses() {
        ArrayList<String> result = new ArrayList<>();
        result.add("apple");
        result.add("aquarium_fish");
        result.add("baby");
        result.add("bear");
        result.add("beaver");
        result.add("bed");
        result.add("bee");
        result.add("beetle");
        result.add("bicycle");
        result.add("bottle");
        result.add("bowl");
        result.add("boy");
        result.add("bridge");
        result.add("bus");
        result.add("butterfly");
        result.add("camel");
        result.add("can");
        result.add("castle");
        result.add("caterpillar");
        result.add("cattle");
        result.add("chair");
        result.add("chimpanzee");
        result.add("clock");
        result.add("cloud");
        result.add("cockroach");
        result.add("couch");
        result.add("crab");
        result.add("crocodile");
        result.add("cup");
        result.add("dinosaur");
        result.add("dolphin");
        result.add("elephant");
        result.add("flatfish");
        result.add("forest");
        result.add("fox");
        result.add("girl");
        result.add("hamster");
        result.add("house");
        result.add("kangaroo");
        result.add("keyboard");
        result.add("lamp");
        result.add("lawn_mower");
        result.add("leopard");
        result.add("lion");
        result.add("lizard");
        result.add("lobster");
        result.add("man");
        result.add("maple_tree");
        result.add("motorcycle");
        result.add("mountain");
        result.add("mouse");
        result.add("mushroom");
        result.add("oak_tree");
        result.add("orange");
        result.add("orchid");
        result.add("otter");
        result.add("palm_tree");
        result.add("pear");
        result.add("pickup_truck");
        result.add("pine_tree");
        result.add("plain");
        result.add("plate");
        result.add("poppy");
        result.add("porcupine");
        result.add("possum");
        result.add("rabbit");
        result.add("raccoon");
        result.add("ray");
        result.add("road");
        result.add("rocket");
        result.add("rose");
        result.add("sea");
        result.add("seal");
        result.add("shark");
        result.add("shrew");
        result.add("skunk");
        result.add("skyscraper");
        result.add("snail");
        result.add("snake");
        result.add("spider");
        result.add("squirrel");
        result.add("streetcar");
        result.add("sunflower");
        result.add("sweet_pepper");
        result.add("table");
        result.add("tank");
        result.add("telephone");
        result.add("television");
        result.add("tiger");
        result.add("tractor");
        result.add("train");
        result.add("trout");
        result.add("tulip");
        result.add("turtle");
        result.add("wardrobe");
        result.add("whale");
        result.add("willow_tree");
        result.add("wolf");
        result.add("woman");
        result.add("worm");
        result.add("Couch");
        result.add("Veggies");
        result.add("Flowers");
        result.add("Fruits");
        result.add("Cars");
        result.add("Trucks");
        result.add("Vans");
        result.add("Computer");
        result.add("Animals");
        return result;
    }

    // Create a list of super classes
    protected ArrayList<String> makeSuperClasses() {

        ArrayList<String> result = new ArrayList<>();
        result.add("aquatic_mammals");
        result.add("fish");
        result.add("flowers");
        result.add("food_containers");
        result.add("fruit_and_vegetables");
        result.add("household_electrical_devices");
        result.add("household_furniture");
        result.add("insects");
        result.add("large_carnivores");
        result.add("large_man-made_outdoor_things");
        result.add("large_natural_outdoor_scenes");
        result.add("large_omnivores_and_herbivores");
        result.add("medium_mammals");
        result.add("non-insect_invertebrates");
        result.add("people");
        result.add("reptiles");
        result.add("small_mammals");
        result.add("trees");
        result.add("vehicles_1");
        result.add("vehicles_2");

        return result;
    }

}

