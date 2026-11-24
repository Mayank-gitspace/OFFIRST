import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:offirst_dr_screening/main.dart';
import 'package:offirst_dr_screening/patient_store.dart';

class _EmptyStore extends PatientStore {
  @override
  Future<List<PatientRecord>> load() async => [];
}

void main() {
  testWidgets('opens offline bilateral screening from the dashboard',
      (tester) async {
    await tester.pumpWidget(OffirstScreeningApp(store: _EmptyStore()));
    await tester.tap(find.text('Start screening'));
    await tester.pumpAndSettle();
    expect(find.text('Retinal image screening'), findsOneWidget);
    expect(find.text('Add fundus images'), findsOneWidget);
    expect(find.text('Right eye (OD)'), findsWidgets);
    expect(find.text('Left eye (OS)'), findsWidgets);
    final save = tester.widget<FilledButton>(
        find.widgetWithText(FilledButton, 'Finalize and save visit'));
    expect(save.onPressed, isNull);
  });
}
