import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:offirst_dr_screening/main.dart';
import 'package:offirst_dr_screening/patient_store.dart';
import 'package:offirst_dr_screening/screening_service.dart';

class _EmptyStore extends PatientStore {
  @override
  Future<List<PatientRecord>> load() async => [];
}

void main() {
  testWidgets('opens offline bilateral screening from the dashboard',
      (tester) async {
    await tester.pumpWidget(OffirstScreeningApp(
        store: _EmptyStore(),
        screeningService: ScreeningService.forTesting(
            initializeModel: () async {},
            runPrediction: (_) async => const ScreeningResult(.1))));
    await tester.tap(find.text('Start screening'));
    // The indexed Experts tab can be loading while Screening is selected, so
    // a perpetual progress animation in an inactive tab must not block settle.
    await tester.pump(const Duration(milliseconds: 500));
    expect(find.text('Retinal image screening'), findsOneWidget);
    expect(find.text('Add fundus images'), findsOneWidget);
    expect(find.text('Right eye (OD)'), findsWidgets);
    expect(find.text('Left eye (OS)'), findsWidgets);
    final saveFinder =
        find.widgetWithText(FilledButton, 'Finalize and save visit');
    await tester.scrollUntilVisible(saveFinder, 160,
        scrollable: find.byType(Scrollable).last);
    final save = tester.widget<FilledButton>(saveFinder);
    expect(save.onPressed, isNull);
  });
}
