extern "C" {
    #include <avr/io.h>
}

#include <Arduino.h>

const uint16_t SERVO_MIN_US = 500;
const uint16_t SERVO_MAX_US = 2500;
const uint16_t SERVO_MAX_ANGLE = 180;

volatile int current_angle = 0;

void servo_init() {
    DDRB |= (1 << PB1);

    TCCR1A = (1 << COM1A1) | (1 << WGM11);
    TCCR1B = (1 << WGM13)  | (1 << WGM12) | (1 << CS11);
    ICR1 = 40000;
}

void servo_write_pulse(int angle) {

    if (angle < 0) angle = 0;
    if (angle > SERVO_MAX_ANGLE) angle = SERVO_MAX_ANGLE;

    uint32_t range_us = SERVO_MAX_US - SERVO_MIN_US;

    uint16_t pulse_us =
        SERVO_MIN_US + (uint32_t)angle * range_us / SERVO_MAX_ANGLE;

    OCR1A = pulse_us * 2;
}

void servo_set_angle(int target_angle, uint16_t speed_ms) {

    if (target_angle < 0)  target_angle = 0;
    if (target_angle > SERVO_MAX_ANGLE) target_angle = SERVO_MAX_ANGLE;

    while (current_angle != target_angle) {

        if (current_angle < target_angle)
            current_angle++;
        else
            current_angle--;

        servo_write_pulse(current_angle);

        delay(speed_ms);
    }
}

void setup() {

    servo_init();

    servo_set_angle(1,   5);   // langsam
    delay(1000);

    servo_set_angle(90,  3);   // mittel
    delay(1000);

    servo_set_angle(180, 2);   // schnell
    delay(1000);

    servo_set_angle(45,  20);   // sehr weich
}

void loop() {
}
