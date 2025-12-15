package edu.jkiryla.aiexplainer;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button btnMnist = findViewById(R.id.btn_mnist);
        btnMnist.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, MnistActivity.class);
                startActivity(intent);
            }
        });

        Button btnFace = findViewById(R.id.btn_face_recognition);
        btnFace.setOnClickListener(v -> {
            Intent intent = new Intent(MainActivity.this, FaceActivity.class);
            startActivity(intent);
        });
    }
}