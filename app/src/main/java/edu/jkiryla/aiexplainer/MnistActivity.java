package edu.jkiryla.aiexplainer;

import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton; // <--- DODANY IMPORT
import android.widget.ImageView;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.MappedByteBuffer;

public class MnistActivity extends AppCompatActivity {

    private DrawView drawView;
    private TextView resultText;
    private ImageView previewImage;
    private Interpreter tflite;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_mnist);

        drawView = findViewById(R.id.draw_view);
        resultText = findViewById(R.id.result_text);
        previewImage = findViewById(R.id.preview_image);

        ImageButton btnBack = findViewById(R.id.btn_back_mnist);
        btnBack.setOnClickListener(v -> finish());

        Button btnClear = findViewById(R.id.btn_clear);
        Button btnClassify = findViewById(R.id.btn_classify);

        try {
            tflite = new Interpreter(loadModelFile());
        } catch (IOException e) {
            e.printStackTrace();
        }

        btnClear.setOnClickListener(v -> {
            drawView.clearCanvas();
            resultText.setText("Narysuj cyfrę (0-9)");
            previewImage.setImageResource(0);
            previewImage.setBackgroundColor(Color.parseColor("#DDDDDD"));
        });

        btnClassify.setOnClickListener(v -> classifyDrawing());
    }

    private void classifyDrawing() {
        if (tflite == null) {
            resultText.setText("Błąd modelu!");
            return;
        }

        Bitmap originalBitmap = drawView.getBitmap();

        Bitmap smallBitmap = Bitmap.createScaledBitmap(originalBitmap, 28, 28, true);

        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(4 * 28 * 28 * 1);
        inputBuffer.order(ByteOrder.nativeOrder());

        Bitmap previewBitmap = Bitmap.createBitmap(28, 28, Bitmap.Config.ARGB_8888);

        int[] pixels = new int[28 * 28];
        smallBitmap.getPixels(pixels, 0, 28, 0, 0, 28, 28);

        for (int i = 0; i < pixels.length; i++) {
            int pixelValue = pixels[i];

            int r = (pixelValue >> 16) & 0xFF;

            float normalized = (255.0f - r) / 255.0f;

            inputBuffer.putFloat(normalized);

            int previewGray = (int) (normalized * 255);
            int previewColor = Color.rgb(previewGray, previewGray, previewGray);
            previewBitmap.setPixel(i % 28, i / 28, previewColor);
        }

        Bitmap displayBitmap = Bitmap.createScaledBitmap(previewBitmap, 140, 140, false);
        previewImage.setImageBitmap(displayBitmap);

        float[][] output = new float[1][10];
        tflite.run(inputBuffer, output);

        int maxIndex = -1;
        float maxProb = 0.0f;
        for (int i = 0; i < 10; i++) {
            if (output[0][i] > maxProb) {
                maxProb = output[0][i];
                maxIndex = i;
            }
        }

        resultText.setText(String.format("Cyfra: %d | Pewność: %.1f%%", maxIndex, maxProb * 100));
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("mnist_model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
}