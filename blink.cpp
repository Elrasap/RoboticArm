#include <avr/io.h>
#include <util/delay.h>

using namespace std;

int main ()
{
	DDRB |= (1<<PB5);

	while (1)
	{
		PORTB = 0xa0;
		_delay_ms (1000);

		PORTB = 0x00;
		_delay_ms (1000);
	}
}
/*#include <arduino>
int ledPin = 13;                 // LED connected to digital pin 13

void setup()
{
  pinMode(ledPin, OUTPUT);      // sets the digital pin as output
}

void loop()
{
  digitalWrite(ledPin, HIGH);   // sets the LED on
  delay(1000);                  // waits for a second
  digitalWrite(ledPin, LOW);    // sets the LED off
  delay(1000);                  // waits for a second
}
*/
