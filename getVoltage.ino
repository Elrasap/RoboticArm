const int hallPin = A0;

void setup() {
  Serial.begin(9600);
}

void loop() {
  int rawValue = analogRead(hallPin);
  float voltage = rawValue * (5.0 / 1023.0);

  Serial.print("Rohwert: ");
  Serial.print(rawValue);
  Serial.print("   Spannung: ");
  Serial.print(voltage);
  Serial.println(" V");

  delay(200);
}

