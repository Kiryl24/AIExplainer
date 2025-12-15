package edu.jkiryla.aiexplainer;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.concurrent.ExecutionException;

public class FaceActivity extends AppCompatActivity {

    private static final int CAMERA_PERMISSION_CODE = 100;
    private static final int MODEL_INPUT_SIZE = 48;

    private PreviewView viewFinder;
    private TextView resultText;
    private ImageView modelInputPreview;
    private Interpreter tflite;

    private Bitmap lastCapturedBitmap = null;

    private final String[] emotions = {
            "Neutralny", "Radość", "Zaskoczenie", "Smutek",
            "Złość", "Obrzydzenie", "Strach", "Pogarda"
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_face);

        viewFinder = findViewById(R.id.viewFinder);
        resultText = findViewById(R.id.text_emotion_result);
        modelInputPreview = findViewById(R.id.model_input_preview);
        Button btnClassify = findViewById(R.id.btn_classify_face);

        ImageButton btnBack = findViewById(R.id.btn_back_face);
        btnBack.setOnClickListener(v -> finish());

        ImageButton btnExplainFace = findViewById(R.id.btn_explain_face);

        btnExplainFace.setOnClickListener(v -> {
            Bitmap bitmapToSend;

            if (lastCapturedBitmap != null) {

                bitmapToSend = lastCapturedBitmap;
            } else {

                Bitmap bitmap = viewFinder.getBitmap();
                if (bitmap == null) {
                    Toast.makeText(this, "Brak obrazu z kamery", Toast.LENGTH_SHORT).show();
                    return;
                }

                int minDimension = Math.min(bitmap.getWidth(), bitmap.getHeight());
                int cropSize = (int) (minDimension * 0.6);
                int startX = (bitmap.getWidth() - cropSize) / 2;
                int startY = (bitmap.getHeight() - cropSize) / 2;
                Bitmap croppedBitmap = Bitmap.createBitmap(bitmap, startX, startY, cropSize, cropSize);
                bitmapToSend = Bitmap.createScaledBitmap(croppedBitmap, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, true);
            }

            java.io.ByteArrayOutputStream stream = new java.io.ByteArrayOutputStream();
            bitmapToSend.compress(Bitmap.CompressFormat.PNG, 100, stream);
            byte[] byteArray = stream.toByteArray();

            Intent intent = new Intent(FaceActivity.this, ExplainActivity.class);
            intent.putExtra("image_data", byteArray);
            intent.putExtra("title", "Explainer: TWARZ");
            startActivity(intent);
        });

        try {
            tflite = new Interpreter(loadModelFile());
        } catch (IOException e) {
            e.printStackTrace();
            resultText.setText("Błąd ładowania modelu!");
        }

        if (checkCameraPermission()) {
            startCamera();
        } else {
            requestCameraPermission();
        }

        btnClassify.setOnClickListener(v -> classifyCurrentFrame());
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(viewFinder.getSurfaceProvider());
                CameraSelector cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA;
                try {
                    cameraProvider.unbindAll();
                    cameraProvider.bindToLifecycle(this, cameraSelector, preview);
                } catch (Exception exc) {
                    Log.e("FaceActivity", "Use case binding failed", exc);
                }
            } catch (ExecutionException | InterruptedException e) {
                Log.e("FaceActivity", "Camera provider failure", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void classifyCurrentFrame() {
        if (tflite == null) {
            Toast.makeText(this, "Model niezaładowany", Toast.LENGTH_SHORT).show();
            return;
        }

        Bitmap bitmap = viewFinder.getBitmap();
        if (bitmap == null) return;

        int minDimension = Math.min(bitmap.getWidth(), bitmap.getHeight());
        int cropSize = (int) (minDimension * 0.6);
        int startX = (bitmap.getWidth() - cropSize) / 2;
        int startY = (bitmap.getHeight() - cropSize) / 2;

        Bitmap croppedBitmap = Bitmap.createBitmap(bitmap, startX, startY, cropSize, cropSize);
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(croppedBitmap, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, true);

        lastCapturedBitmap = scaledBitmap;

        modelInputPreview.setImageBitmap(scaledBitmap);
        ByteBuffer inputBuffer = convertBitmapToGrayscaleByteBuffer(scaledBitmap);

        float[][] output = new float[1][8];
        tflite.run(inputBuffer, output);

        float[] probabilities = softmax(output[0]);

        int bestIndex = -1;
        float maxProb = 0.0f;
        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                bestIndex = i;
            }
        }

        if (bestIndex != -1 && bestIndex < emotions.length) {
            resultText.setText(String.format("%s (%.1f%%)", emotions[bestIndex], maxProb * 100));
            if (emotions[bestIndex].equals("Radość")) resultText.setTextColor(Color.parseColor("#2ECC71"));
            else if (emotions[bestIndex].equals("Złość")) resultText.setTextColor(Color.RED);
            else resultText.setTextColor(Color.DKGRAY);
        } else {
            resultText.setText("Nie rozpoznano");
        }
    }

    private float[] softmax(float[] logits) {
        float[] probs = new float[logits.length];
        float maxLogit = -Float.MAX_VALUE;
        for (float val : logits) { if (val > maxLogit) maxLogit = val; }
        float sum = 0.0f;
        for (int i = 0; i < logits.length; i++) {
            probs[i] = (float) Math.exp(logits[i] - maxLogit);
            sum += probs[i];
        }
        for (int i = 0; i < logits.length; i++) { probs[i] /= sum; }
        return probs;
    }

    private ByteBuffer convertBitmapToGrayscaleByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 1);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] pixels = new int[MODEL_INPUT_SIZE * MODEL_INPUT_SIZE];
        bitmap.getPixels(pixels, 0, MODEL_INPUT_SIZE, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
        for (int pixelValue : pixels) {
            int r = (pixelValue >> 16) & 0xFF;
            float gray = (r * 0.299f + (pixelValue >> 8 & 0xFF) * 0.587f + (pixelValue & 0xFF) * 0.114f);
            byteBuffer.putFloat(gray / 255.0f);
        }
        return byteBuffer;
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("ferplus_model_pd_best.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private boolean checkCameraPermission() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }

    private void requestCameraPermission() {
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_CODE);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera();
            } else {
                Toast.makeText(this, "Wymagane uprawnienia do kamery", Toast.LENGTH_SHORT).show();
            }
        }
    }
}