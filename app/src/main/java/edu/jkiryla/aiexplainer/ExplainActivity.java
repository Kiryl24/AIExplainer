package edu.jkiryla.aiexplainer;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import java.util.ArrayList;
import java.util.List;

public class ExplainActivity extends AppCompatActivity {

    private static class LayerData {
        String name;
        Bitmap image;
        String description;

        LayerData(String name, Bitmap image, String description) {
            this.name = name;
            this.image = image;
            this.description = description;
        }
    }

    private List<LayerData> layers = new ArrayList<>();
    private int currentLayerIndex = 0;

    private long lastToastTime = 0;
    private static final int TOAST_COOLDOWN = 5000; // 5 sekund


    private TextView tvLayerName, tvLayerDesc, tvIndicator;
    private ImageView ivLayerVisual;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_explain);

        tvLayerName = findViewById(R.id.tv_layer_name);
        tvLayerDesc = findViewById(R.id.tv_layer_desc);
        tvIndicator = findViewById(R.id.tv_layer_indicator);
        ivLayerVisual = findViewById(R.id.iv_layer_visual);

        ImageButton btnBack = findViewById(R.id.btn_back_explain);
        ImageButton btnPrev = findViewById(R.id.btn_prev_layer);
        ImageButton btnNext = findViewById(R.id.btn_next_layer);
        TextView tvTitle = findViewById(R.id.tv_explain_title);

        String title = getIntent().getStringExtra("title");
        if (title != null) tvTitle.setText(title);

        byte[] byteArray = getIntent().getByteArrayExtra("image_data");
        if (byteArray != null) {
            Bitmap originalBitmap = BitmapFactory.decodeByteArray(byteArray, 0, byteArray.length);
            generateLayers(originalBitmap);
        }

        updateUI();

        btnBack.setOnClickListener(v -> finish());

        btnPrev.setOnClickListener(v -> {
            if (currentLayerIndex > 0) {
                currentLayerIndex--;
                updateUI();
            } else {
                showToastWithCooldown("To jest Warstwa wejściowa!");
            }
        });

        btnNext.setOnClickListener(v -> {
            if (currentLayerIndex < layers.size() - 1) {
                currentLayerIndex++;
                updateUI();
            } else {
                showToastWithCooldown("To jest Warstwa wyjściowa!");
            }
        });
    }

    private void showToastWithCooldown(String message) {
        long currentTime = System.currentTimeMillis();

        if (currentTime - lastToastTime < TOAST_COOLDOWN) {
            return;
        }

        Toast.makeText(this, message, Toast.LENGTH_SHORT).show();
        lastToastTime = currentTime;
    }

    private void updateUI() {
        if (layers.isEmpty()) return;

        LayerData currentData = layers.get(currentLayerIndex);

        tvLayerName.setText(currentData.name);
        tvLayerDesc.setText(currentData.description);

        Bitmap displayBitmap = Bitmap.createScaledBitmap(currentData.image, 300, 300, false);
        ivLayerVisual.setImageBitmap(displayBitmap);

        tvIndicator.setText((currentLayerIndex + 1) + " / " + layers.size());
    }

    private void generateLayers(Bitmap inputBitmap) {
        layers.add(new LayerData(
                "Warstwa wejściowa",
                inputBitmap,
                "To surowe dane, które widzi model. Każdy piksel ma wartość liczbową odpowiadającą jasności."
        ));

        float[][] verticalFilter = {
                {-1, 0, 1},
                {-2, 0, 2},
                {-1, 0, 1}
        };
        Bitmap verticalMap = applyConvolution(inputBitmap, verticalFilter);
        layers.add(new LayerData(
                "Warstwa 1",
                verticalMap,
                "Wykrywanie cech pionowych. Neurony w tej warstwie aktywują się tam, gdzie widzą pionowe krawędzie."
        ));

        float[][] horizontalFilter = {
                {-1, -2, -1},
                { 0,  0,  0},
                { 1,  2,  1}
        };
        Bitmap horizontalMap = applyConvolution(inputBitmap, horizontalFilter);
        layers.add(new LayerData(
                "Warstwa 2",
                horizontalMap,
                "Wykrywanie cech poziomych. Neurony reagują na górne i dolne krawędzie obiektu."
        ));

        Bitmap combinedMap = combineBitmaps(verticalMap, horizontalMap);
        layers.add(new LayerData(
                "Warstwa wyjściowa",
                combinedMap,
                "Model łączy wykryte proste cechy (linie) w bardziej złożone kształty, co pozwala mu podjąć decyzję."
        ));
    }

    private Bitmap applyConvolution(Bitmap src, float[][] kernel) {
        int width = src.getWidth();
        int height = src.getHeight();
        Bitmap result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);

        int[] pixels = new int[width * height];
        src.getPixels(pixels, 0, width, 0, 0, width, height);
        int[] newPixels = new int[width * height];

        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                float sum = 0.0f;

                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int pixel = pixels[(y + ky) * width + (x + kx)];
                        int val = (pixel >> 16) & 0xFF; // Pobranie kanału R (obraz czarno-biały)
                        sum += val * kernel[ky + 1][kx + 1];
                    }
                }

                int finalVal = (int) Math.abs(sum);
                if (finalVal > 255) finalVal = 255;

                newPixels[y * width + x] = Color.rgb(finalVal, finalVal, finalVal);
            }
        }
        result.setPixels(newPixels, 0, width, 0, 0, width, height);
        return result;
    }

    private Bitmap combineBitmaps(Bitmap b1, Bitmap b2) {
        int width = b1.getWidth();
        int height = b1.getHeight();
        Bitmap result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);

        int[] p1 = new int[width*height];
        int[] p2 = new int[width*height];
        int[] out = new int[width*height];

        b1.getPixels(p1, 0, width, 0, 0, width, height);
        b2.getPixels(p2, 0, width, 0, 0, width, height);

        for(int i=0; i<p1.length; i++) {
            int v1 = (p1[i] >> 16) & 0xFF;
            int v2 = (p2[i] >> 16) & 0xFF;
            int max = Math.max(v1, v2);
            out[i] = Color.rgb(max, max, max);
        }
        result.setPixels(out, 0, width, 0, 0, width, height);
        return result;
    }
}