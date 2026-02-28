String str;

void setup ()
{
  Serial.begin(9600);
}

void loop ()
{
  char ch;
  while (Serial.available() > 0) {
    if ('\n' == (ch = Serial.read ())) {
      Serial.println (str);
      str = "";
    } else {
      str += ch;
    }
  }
}