import 'package:flutter/material.dart';

abstract final class OffirstColors {
  static const cobalt = Color(0xFF003399);
  static const cyan = Color(0xFF0070E0);
  static const background = Color(0xFFFFFFFF);
  static const surface = Color(0xFFF4F6F9);
  static const border = Color(0xFFDCE1E7);
  static const text = Color(0xFF111827);
  static const secondaryText = Color(0xFF5A6577);
  static const viewport = Color(0xFF05070A);
  static const charcoal = Color(0xFF12161F);
  static const success = Color(0xFF15803D);
  static const attention = Color(0xFFB45309);
  static const error = Color(0xFFB91C1C);
}

ThemeData buildOffirstTheme() {
  final scheme = ColorScheme.fromSeed(
    seedColor: OffirstColors.cobalt,
    primary: OffirstColors.cobalt,
    secondary: OffirstColors.cyan,
    surface: OffirstColors.background,
    error: OffirstColors.error,
  );
  return ThemeData(
    useMaterial3: true,
    colorScheme: scheme,
    scaffoldBackgroundColor: OffirstColors.background,
    appBarTheme: const AppBarTheme(
      backgroundColor: OffirstColors.background,
      foregroundColor: OffirstColors.text,
      surfaceTintColor: Colors.transparent,
      centerTitle: false,
    ),
    cardTheme: CardThemeData(
      color: OffirstColors.background,
      elevation: 0,
      margin: EdgeInsets.zero,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(8),
        side: const BorderSide(color: OffirstColors.border),
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: OffirstColors.surface,
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(8),
        borderSide: const BorderSide(color: OffirstColors.border),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(8),
        borderSide: const BorderSide(color: OffirstColors.border),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(8),
        borderSide: const BorderSide(color: OffirstColors.cobalt, width: 1.5),
      ),
    ),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        minimumSize: const Size(48, 50),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      ),
    ),
  );
}
