import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'expert_page.dart';
import 'offirst_theme.dart';
import 'onboarding_tour.dart';
import 'patient_store.dart';
import 'screening_page.dart';
import 'screening_service.dart';
import 'visit_page.dart';

void main() => runApp(const OffirstScreeningApp());

class OffirstScreeningApp extends StatelessWidget {
  const OffirstScreeningApp(
      {super.key, this.store, this.screeningService, this.onboarding});
  final ScreeningService? screeningService;
  final PatientStore? store;
  final OnboardingStore? onboarding;

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'OFFIRST',
        debugShowCheckedModeBanner: false,
        theme: buildOffirstTheme(),
        home: OffirstHome(
          store: store ?? PatientStore(),
          service: screeningService,
          onboarding: onboarding ?? FileOnboardingStore(),
        ),
      );
}

class OffirstHome extends StatefulWidget {
  const OffirstHome(
      {super.key, required this.store, this.service, required this.onboarding});
  final ScreeningService? service;
  final PatientStore store;
  final OnboardingStore onboarding;

  @override
  State<OffirstHome> createState() => _OffirstHomeState();
}

class _OffirstHomeState extends State<OffirstHome> {
  late final ScreeningService _service = widget.service ?? ScreeningService();
  @override
  void initState() {
    super.initState();
    _patients = _trackPatients(_patients);
    _service
        .initialize()
        .catchError((Object _) {}); // Service exposes sanitized error state.
    unawaited(_loadOnboarding());
  }

  @override
  void dispose() {
    _tourNotifier.dispose();
    _service.dispose();
    super.dispose();
  }

  int _tab = 0;
  int? _tourStep;
  final _tourNotifier = ValueNotifier<int?>(null);
  final _navigationKey = GlobalKey();
  final _registerKey = GlobalKey();
  final _patientKey = GlobalKey();
  final _modeKey = GlobalKey();
  final _imagesKey = GlobalKey();
  final _saveKey = GlobalKey();
  final _recentKey = GlobalKey();
  final _expertKey = GlobalKey();
  final _expertTile = ExpansionTileController();
  String _query = '';
  PatientStore get _store => widget.store;
  late Future<List<PatientRecord>> _patients = _store.load();
  List<PatientRecord> _records = const [];
  late Future<List<ExpertRecord>> _experts = _store.loadExperts();

  void _refresh() => setState(() {
        _patients = _trackPatients(_store.load());
        _experts = _store.loadExperts();
      });

  Future<List<PatientRecord>> _trackPatients(
      Future<List<PatientRecord>> future) {
    future.then((rows) {
      if (!mounted) return;
      setState(() => _records = rows);
      if (_tourStep != null) _revealTour();
    });
    return future;
  }

  Future<void> _loadOnboarding() async {
    try {
      final done = await widget.onboarding.isComplete();
      if (!mounted || done) return;
      _setTour(0);
    } catch (_) {
      // Leave the tour hidden if the completion flag cannot be read.
    }
  }

  void _setTour(int? step, {int? tab, bool reveal = true}) {
    setState(() {
      _tourStep = step;
      if (tab != null) _tab = tab;
    });
    _tourNotifier.value = step;
    if (step != null && reveal) _revealTour();
  }

  bool _nextEnabled(OnboardingStep step) {
    if (step.needsPatient && _records.isEmpty) return false;
    if (step.needsVisit && !_records.any((patient) => patient.visits.isNotEmpty)) {
      return false;
    }
    if (step.opensVisit && _latestVisit() == null) return false;
    return true;
  }

  void _nextTourStep() {
    final index = _tourStep;
    if (index == null) return;
    final step = onboardingSteps[index];
    if (!_nextEnabled(step)) return;
    if (index >= onboardingSteps.length - 1) {
      unawaited(_finishTour());
      return;
    }
    final next = index + 1;
    if (step.opensVisit) {
      final latest = _latestVisit();
      if (latest == null) return;
      _openVisit(latest.patient, latest.visit);
      return;
    }
    _setTour(next, tab: onboardingSteps[next].tab);
  }

  void _revealTour() {
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      if (!mounted || _tourStep == null) return;
      final step = onboardingSteps[_tourStep!];
      if (step.expandsExpert) {
        try {
          if (!_expertTile.isExpanded) _expertTile.expand();
        } catch (_) {}
        await Future<void>.delayed(const Duration(milliseconds: 250));
      }
      if (!mounted) return;
      final target = _keyFor(step)?.currentContext;
      if (target != null) {
        await Scrollable.ensureVisible(target,
            alignment: 0.3, duration: const Duration(milliseconds: 200));
      }
      if (mounted) setState(() {});
    });
  }

  GlobalKey? _keyFor(OnboardingStep step) => switch (step.target) {
        TourTarget.register => _registerKey,
        TourTarget.patient => _patientKey,
        TourTarget.mode => _modeKey,
        TourTarget.images => _imagesKey,
        TourTarget.save => _saveKey,
        TourTarget.recent => _recentKey,
        TourTarget.expert => _expertKey,
        TourTarget.pdf || null => null,
      };

  Eye? _focusedEye() {
    final step = _tourStep;
    if (step == null) return null;
    return switch (onboardingSteps[step].eye) {
      TourEye.right => Eye.right,
      TourEye.left => Eye.left,
      null => null,
    };
  }

  ({PatientRecord patient, VisitRecord visit})? _latestVisit() {
    PatientRecord? patient;
    VisitRecord? visit;
    for (final record in _records) {
      for (final item in record.visits) {
        if (visit == null || item.date.isAfter(visit.date)) {
          patient = record;
          visit = item;
        }
      }
    }
    if (patient == null || visit == null) return null;
    return (patient: patient, visit: visit);
  }

  String? _tourPatientId() => _latestVisit()?.patient.id ??
      (_records.isEmpty ? null : _records.first.id);

  void _openVisit(PatientRecord patient, VisitRecord visit) {
    final fromTour =
        _tourStep != null && onboardingSteps[_tourStep!].opensVisit;
    if (fromTour) _setTour(tourIndex(TourTarget.pdf), reveal: false);
    unawaited(_pushVisit(patient, visit));
  }

  Future<void> _pushVisit(PatientRecord patient, VisitRecord visit) async {
    final pdf = tourIndex(TourTarget.pdf);
    final recent =
        onboardingSteps.indexWhere((step) => step.opensVisit);
    await Navigator.of(context).push(MaterialPageRoute<void>(
      builder: (_) => VisitPage(
        patient: patient,
        visit: visit,
        store: _store,
        tourStep: _tourNotifier,
        onTourNext: _finishReportCoach,
        onTourSkip: () => unawaited(_finishTour()),
      ),
    ));
    if (!mounted) return;
    if (_tourStep == pdf) _setTour(recent);
  }

  void _finishReportCoach() {
    final expert = tourIndex(TourTarget.expert);
    _setTour(expert, tab: onboardingSteps[expert].tab);
    if (Navigator.of(context).canPop()) Navigator.of(context).pop();
  }

  Future<void> _finishTour() async {
    try {
      await widget.onboarding.markComplete();
    } catch (_) {}
    if (mounted) _setTour(null, reveal: false);
  }

  Future<void> _editPatient([PatientRecord? existing]) async {
    final name = TextEditingController(text: existing?.name ?? '');
    final age = TextEditingController(text: existing?.age?.toString() ?? '');
    final notes = TextEditingController(text: existing?.notes ?? '');
    final formKey = GlobalKey<FormState>();
    final save = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: Text(existing == null ? 'Register patient' : 'Edit patient'),
        content: Form(
          key: formKey,
          child: SingleChildScrollView(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 420),
              child: Column(mainAxisSize: MainAxisSize.min, children: [
                if (existing != null)
                  Align(
                    alignment: Alignment.centerLeft,
                    child: Padding(
                      padding: const EdgeInsets.only(bottom: 12),
                      child: Text('Patient ID  ${existing.id}'),
                    ),
                  ),
                TextFormField(
                  controller: name,
                  autofocus: true,
                  textCapitalization: TextCapitalization.words,
                  decoration: const InputDecoration(labelText: 'Full name *'),
                  validator: (value) => value == null || value.trim().isEmpty
                      ? 'Enter a patient name.'
                      : null,
                ),
                const SizedBox(height: 12),
                TextFormField(
                  controller: age,
                  keyboardType: TextInputType.number,
                  decoration:
                      const InputDecoration(labelText: 'Age (optional)'),
                  validator: (value) {
                    if (value == null || value.trim().isEmpty) return null;
                    final parsed = int.tryParse(value.trim());
                    return parsed == null || parsed < 0 || parsed > 130
                        ? 'Enter an age from 0 to 130.'
                        : null;
                  },
                ),
                const SizedBox(height: 12),
                TextFormField(
                  controller: notes,
                  maxLines: 3,
                  decoration: const InputDecoration(
                    labelText: 'Notes (optional)',
                    helperText: 'Stored only on this device.',
                  ),
                ),
              ]),
            ),
          ),
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(dialogContext, false),
              child: const Text('Cancel')),
          FilledButton(
              onPressed: () {
                if (formKey.currentState!.validate()) {
                  Navigator.pop(dialogContext, true);
                }
              },
              child: const Text('Save')),
        ],
      ),
    );
    if (save != true || !mounted) return;
    final rawAge = age.text.trim();
    final record = PatientRecord(
      id: existing?.id ?? recordId('OFF'),
      name: name.text.trim(),
      createdAt: existing?.createdAt ?? DateTime.now().toUtc(),
      age: rawAge.isEmpty ? null : int.tryParse(rawAge),
      notes: notes.text.trim().isEmpty ? null : notes.text.trim(),
      expertIds: existing?.expertIds,
      visits: existing?.visits,
    );
    try {
      if (existing == null) {
        await _store.addPatient(record);
      } else {
        await _store.updatePatient(record);
      }
      if (mounted) _refresh();
    } catch (_) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Could not save the patient record.')),
        );
      }
    }
  }

  Future<void> _deletePatient(PatientRecord patient) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        icon: const Icon(Icons.delete_outline),
        title: const Text('Delete patient record?'),
        content: Text(
            'Delete ${patient.name} (${patient.id}) and its ${patient.visits.length} saved visit(s) and local images? This cannot be undone.'),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(dialogContext, false),
              child: const Text('Cancel')),
          FilledButton.tonal(
              onPressed: () => Navigator.pop(dialogContext, true),
              child: const Text('Delete patient')),
        ],
      ),
    );
    if (confirmed != true || !mounted) return;
    try {
      await _store.deletePatient(patient.id);
      if (mounted) {
        _refresh();
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Patient record deleted.')),
        );
      }
    } catch (_) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
          content: Text('Could not completely delete the patient record.'),
        ));
      }
    }
  }

  Future<void> _assignExperts(PatientRecord patient) async {
    final experts = await _experts;
    if (!mounted) return;
    final selected = Set<String>.of(patient.expertIds);
    final save = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => StatefulBuilder(
        builder: (context, update) => AlertDialog(
          title: Text('Associated experts · ${patient.name}'),
          content: SizedBox(
            width: 420,
            child: experts.isEmpty
                ? const Text('Add an expert record first.')
                : ListView(shrinkWrap: true, children: [
                    for (final expert in experts)
                      CheckboxListTile(
                        value: selected.contains(expert.id),
                        title: Text(expert.name),
                        subtitle: Text(expert.specialty ?? expert.id),
                        onChanged: (value) => update(() {
                          if (value == true) {
                            selected.add(expert.id);
                          } else {
                            selected.remove(expert.id);
                          }
                        }),
                      ),
                  ]),
          ),
          actions: [
            TextButton(
                onPressed: () => Navigator.pop(dialogContext, false),
                child: const Text('Cancel')),
            FilledButton(
                onPressed: () => Navigator.pop(dialogContext, true),
                child: const Text('Save')),
          ],
        ),
      ),
    );
    if (save != true || !mounted) return;
    try {
      await _store.setPatientExperts(patient.id, selected.toList());
      if (mounted) _refresh();
    } catch (_) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
            content: Text('Could not update the expert association.')));
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final tourStep = _tourStep;
    final current = tourStep == null ? null : onboardingSteps[tourStep];
    return Stack(children: [
      Scaffold(
        appBar: AppBar(
          title: const Text('OFFIRST'),
          actions: [
            ListenableBuilder(
              listenable: _service,
              builder: (context, _) => ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 220),
                child: TextButton.icon(
                  icon: Icon(
                      _service.readiness == ModelReadiness.ready
                          ? Icons.memory_outlined
                          : Icons.info_outline,
                      size: 18),
                  label: Text(
                      switch (_service.readiness) {
                        ModelReadiness.initializing => 'Initializing model',
                        ModelReadiness.ready => useExperimentalV21
                            ? 'Model ready — OFFIRST v2.1'
                            : 'Model ready — OFFIRST v1.1',
                        ModelReadiness.unavailable =>
                          'Model unavailable — details',
                      },
                      softWrap: true),
                  onPressed: () => showDialog<void>(
                      context: context,
                      builder: (context) => AlertDialog(
                            title: const Text('On-device model status'),
                            content: Text(_service
                                    .initializationError?.message ??
                                (_service.readiness == ModelReadiness.ready
                                    ? 'ONNX session initialized and input/output contract verified. This status does not indicate that an image has been screened.'
                                    : 'Loading the packaged model and checking its input/output contract.')),
                            actions: [
                              if (_service.readiness ==
                                  ModelReadiness.unavailable)
                                TextButton(
                                    onPressed: () {
                                      Navigator.pop(context);
                                      _service
                                          .initialize()
                                          .catchError((Object _) {});
                                    },
                                    child: const Text('Retry model loading')),
                              TextButton(
                                  onPressed: () => Navigator.pop(context),
                                  child: const Text('Close')),
                            ],
                          )),
                ),
              ),
            ),
          ],
        ),
        body: IndexedStack(index: _tab, children: [
          _dashboard(),
          _patientList(),
          ScreeningPage(
              store: _store,
              patients: _patients,
              onSaved: _refresh,
              service: _service,
              patientKey: _patientKey,
              modeKey: _modeKey,
              imagesKey: _imagesKey,
              saveKey: _saveKey,
              focusedEye: _focusedEye()),
          ExpertDirectoryPage(store: _store, onChanged: _refresh),
          const _AboutPage(),
        ]),
        bottomNavigationBar: NavigationBar(
          key: _navigationKey,
          selectedIndex: _tab,
          onDestinationSelected: (value) => setState(() => _tab = value),
          destinations: const [
            NavigationDestination(
                icon: Icon(Icons.home_outlined), label: 'Home'),
            NavigationDestination(
                icon: Icon(Icons.people_outline), label: 'Patients'),
            NavigationDestination(
                icon: Icon(Icons.remove_red_eye_outlined), label: 'Screening'),
            NavigationDestination(
                icon: Icon(Icons.medical_services_outlined), label: 'Experts'),
            NavigationDestination(
                icon: Icon(Icons.info_outline), label: 'About'),
          ],
        ),
      ),
      if (current != null && current.target != TourTarget.pdf)
        OnboardingOverlay(
          label: current.label,
          message: current.message,
          index: tourStep!,
          count: onboardingSteps.length,
          last: tourStep == onboardingSteps.length - 1,
          nextEnabled: _nextEnabled(current),
          targetKey: _keyFor(current),
          navigationKey: _navigationKey,
          navigationSlot: current.navSlot,
          passHole: current.passHole,
          allowPage: current.allowPage,
          onNext: _nextTourStep,
          onSkip: () => unawaited(_finishTour()),
        ),
    ]);
  }

  Widget _dashboard() => FutureBuilder<List<PatientRecord>>(
        future: _patients,
        builder: (context, snapshot) {
          final patients = snapshot.data ?? <PatientRecord>[];
          final visits = patients.fold<int>(0, (n, p) => n + p.visits.length);
          final images = patients.fold<int>(
              0,
              (n, p) =>
                  n +
                  p.visits.fold<int>(0, (m, visit) => m + visit.images.length));
          final recent = patients
              .expand((patient) => patient.visits
                  .map((visit) => (patient: patient, visit: visit)))
              .toList()
            ..sort((a, b) => b.visit.date.compareTo(a.visit.date));
          return ListView(padding: const EdgeInsets.all(20), children: [
            Text('FIELD SCREENING',
                style: Theme.of(context).textTheme.labelLarge?.copyWith(
                    color: OffirstColors.cobalt, letterSpacing: 1.4)),
            const SizedBox(height: 8),
            Text('Retinal screening',
                style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                    color: OffirstColors.text, fontWeight: FontWeight.w700)),
            Text(
                'Experimental diabetic retinopathy screening powered by on-device AI.',
                style: Theme.of(context)
                    .textTheme
                    .bodyLarge
                    ?.copyWith(color: OffirstColors.secondaryText)),
            const SizedBox(height: 20),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('ON-DEVICE SCREENING',
                          style: Theme.of(context)
                              .textTheme
                              .labelLarge
                              ?.copyWith(
                                  color: OffirstColors.cobalt,
                                  letterSpacing: 1.2)),
                      const SizedBox(height: 8),
                      const Text(
                          'Select a patient and add retinal images. Inference runs locally on this device.',
                          style: TextStyle(fontSize: 17, height: 1.4)),
                      const SizedBox(height: 14),
                      SizedBox(
                        width: double.infinity,
                        child: FilledButton.icon(
                          onPressed: () => setState(() => _tab = 2),
                          icon: const Icon(Icons.remove_red_eye_outlined),
                          label: const Text('Start screening'),
                        ),
                      ),
                      TextButton.icon(
                          onPressed: _editPatient,
                          icon: const Icon(Icons.person_add_alt_outlined),
                          label: const Text('Register patient')),
                    ]),
              ),
            ),
            const SizedBox(height: 12),
            Row(children: [
              Expanded(child: _metric('${patients.length}', 'Patients')),
              const SizedBox(width: 8),
              Expanded(child: _metric('$visits', 'Visits')),
              const SizedBox(width: 8),
              Expanded(child: _metric('$images', 'Images')),
            ]),
            const SizedBox(height: 24),
            Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
              Text('Recent visits',
                  style: Theme.of(context).textTheme.titleLarge),
              TextButton(
                  onPressed: () => setState(() => _tab = 1),
                  child: const Text('Patients')),
            ]),
            if (recent.isEmpty)
              Card(
                key: _recentKey,
                child: const Padding(
                    padding: EdgeInsets.all(18),
                    child: Text('Saved screening visits will appear here.')),
              )
            else
              ...recent.take(4).toList().asMap().entries.map((slot) {
                final entry = slot.value;
                return Card(
                  key: slot.key == 0 ? _recentKey : null,
                  child: ListTile(
                    leading: const Icon(Icons.history,
                        color: OffirstColors.cobalt),
                    title: Text(entry.patient.name),
                    subtitle: Text(
                        '${MaterialLocalizations.of(context).formatMediumDate(entry.visit.date.toLocal())} · ${entry.visit.images.length} image(s)'),
                    trailing: const Icon(Icons.chevron_right),
                    onTap: () => _openVisit(entry.patient, entry.visit),
                  ),
                );
              }),
            const SizedBox(height: 20),
            const _PrivacyNotice(),
          ]);
        },
      );

  Widget _metric(String value, String label) => Card(
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 8),
          child: Column(children: [
            Text(value,
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                    color: OffirstColors.text, fontWeight: FontWeight.w700)),
            Text(label,
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.labelMedium),
          ]),
        ),
      );

  Widget _patientList() => FutureBuilder<List<PatientRecord>>(
        future: _patients,
        builder: (context, snapshot) {
          if (snapshot.connectionState != ConnectionState.done) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snapshot.hasError) {
            return const Center(child: Text('Could not load patient records.'));
          }
          final query = _query.trim().toLowerCase();
          final rows = (snapshot.data ?? <PatientRecord>[])
              .where((p) =>
                  p.name.toLowerCase().contains(query) ||
                  p.id.toLowerCase().contains(query))
              .toList();
          return ListView(padding: const EdgeInsets.all(16), children: [
            Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
              Text('Patients',
                  style: Theme.of(context).textTheme.headlineSmall),
              IconButton(
                  key: _registerKey,
                  tooltip: 'Register patient',
                  onPressed: _editPatient,
                  icon: const Icon(Icons.person_add_alt_outlined)),
            ]),
            const SizedBox(height: 10),
            TextField(
                onChanged: (value) => setState(() => _query = value),
                decoration: const InputDecoration(
                    prefixIcon: Icon(Icons.search),
                    hintText: 'Search name or patient ID')),
            const SizedBox(height: 12),
            if (rows.isEmpty)
              const Padding(
                  padding: EdgeInsets.all(24),
                  child: Text('No matching local patient records.')),
            ...rows.map((patient) {
              final guided = patient.id == _tourPatientId();
              return Card(
                  child: ExpansionTile(
                    key: ValueKey(patient.id),
                    controller: guided ? _expertTile : null,
                    leading: const CircleAvatar(child: Icon(Icons.person)),
                    title: Text(patient.name),
                    subtitle: Text(
                        '${patient.id} · ${patient.visits.length} saved visits'),
                    trailing: PopupMenuButton<String>(
                      tooltip: 'Patient actions',
                      onSelected: (action) {
                        if (action == 'edit') _editPatient(patient);
                        if (action == 'delete') _deletePatient(patient);
                      },
                      itemBuilder: (_) => const [
                        PopupMenuItem(value: 'edit', child: Text('Edit')),
                        PopupMenuItem(value: 'delete', child: Text('Delete')),
                      ],
                    ),
                    children: [
                      FutureBuilder<List<ExpertRecord>>(
                        future: _experts,
                        builder: (context, expertSnapshot) {
                          final assigned = (expertSnapshot.data ?? [])
                              .where((expert) =>
                                  patient.expertIds.contains(expert.id))
                              .map((expert) => expert.name)
                              .toList();
                          return KeyedSubtree(
                            key: guided ? _expertKey : null,
                            child: ListTile(
                              title: const Text('Locally associated experts'),
                              subtitle: Text(assigned.isEmpty
                                  ? 'None recorded'
                                  : assigned.join(', ')),
                              trailing: IconButton(
                                tooltip: 'Manage expert associations',
                                onPressed: () => _assignExperts(patient),
                                icon: const Icon(Icons.edit_outlined),
                              ),
                            ),
                          );
                        },
                      ),
                      ListTile(
                        title: const Text('Patient ID'),
                        subtitle: SelectableText(patient.id),
                        trailing: IconButton(
                          tooltip: 'Copy patient ID',
                          icon: const Icon(Icons.copy_outlined),
                          onPressed: () async {
                            await Clipboard.setData(
                                ClipboardData(text: patient.id));
                            if (context.mounted) {
                              ScaffoldMessenger.of(context).showSnackBar(
                                  const SnackBar(
                                      content: Text('Patient ID copied.')));
                            }
                          },
                        ),
                      ),
                      if (patient.age != null)
                        ListTile(title: Text('Age ${patient.age}')),
                      if (patient.notes?.isNotEmpty == true)
                        ListTile(
                            title: const Text('Notes'),
                            subtitle: Text(patient.notes!)),
                      ...patient.visits.map((visit) => ListTile(
                            leading: const Icon(Icons.history),
                            title: Text(MaterialLocalizations.of(context)
                                .formatMediumDate(visit.date.toLocal())),
                            subtitle: Text(
                                'OD: ${visit.forEye(Eye.right).length} · OS: ${visit.forEye(Eye.left).length}'),
                            trailing: const Icon(Icons.chevron_right),
                            onTap: () => Navigator.of(context).push(
                              MaterialPageRoute<void>(
                                builder: (_) => VisitPage(
                                    patient: patient,
                                    visit: visit,
                                    store: _store),
                              ),
                            ),
                          )),
                    ],
                  ),
                );
            }),
          ]);
        },
      );
}

class _AboutPage extends StatelessWidget {
  const _AboutPage();

  @override
  Widget build(BuildContext context) => ListView(
        padding: const EdgeInsets.all(24),
        children: [
          const Icon(Icons.remove_red_eye_outlined,
              size: 42, color: OffirstColors.cobalt),
          const SizedBox(height: 12),
          Text('About OFFIRST',
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.headlineSmall),
          const SizedBox(height: 20),
          const _PrivacyNotice(),
          const SizedBox(height: 12),
          const Card(
            child: Padding(
              padding: EdgeInsets.all(16),
              child: Text(useExperimentalV21
                  ? 'Active model: OFFIRST v2.1 experimental screening. The default result is a 4-class diabetic retinopathy grade on the on-device embedding. Images and records stay on this device.'
                  : 'The app uses the packaged OFFIRST v1.1 ONNX classifier with a fixed 0.48 threshold. It does not upload images or records and does not learn from patient images.'),
            ),
          ),
        ],
      );
}

class _PrivacyNotice extends StatelessWidget {
  const _PrivacyNotice();

  @override
  Widget build(BuildContext context) => const Card(
        child: Padding(
          padding: EdgeInsets.all(16),
          child: Text(
              'OFFIRST is an experimental academic prototype, not a certified medical device. Model results are not clinically validated and must not be used independently for diagnosis or care decisions. Records and images are stored locally in app-private storage without application-level encryption.'),
        ),
      );
}
