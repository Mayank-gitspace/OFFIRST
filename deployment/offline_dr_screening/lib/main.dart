import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'offirst_theme.dart';
import 'expert_page.dart';
import 'patient_store.dart';
import 'screening_page.dart';
import 'screening_service.dart';
import 'visit_page.dart';

void main() => runApp(const OffirstScreeningApp());

class OffirstScreeningApp extends StatelessWidget {
  const OffirstScreeningApp({super.key, this.store, this.screeningService});
  final ScreeningService? screeningService;
  final PatientStore? store;

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'OFFIRST',
        debugShowCheckedModeBanner: false,
        theme: buildOffirstTheme(),
        home: OffirstHome(
            store: store ?? PatientStore(), service: screeningService),
      );
}

class OffirstHome extends StatefulWidget {
  const OffirstHome({super.key, required this.store, this.service});
  final ScreeningService? service;
  final PatientStore store;

  @override
  State<OffirstHome> createState() => _OffirstHomeState();
}

class _OffirstHomeState extends State<OffirstHome> {
  late final ScreeningService _service = widget.service ?? ScreeningService();
  @override
  void initState() {
    super.initState();
    _service
        .initialize()
        .catchError((Object _) {}); // Service exposes sanitized error state.
  }

  @override
  void dispose() {
    _service.dispose();
    super.dispose();
  }

  int _tab = 0;
  String _query = '';
  PatientStore get _store => widget.store;
  late Future<List<PatientRecord>> _patients = _store.load();
  late Future<List<ExpertRecord>> _experts = _store.loadExperts();

  void _refresh() => setState(() {
        _patients = _store.load();
        _experts = _store.loadExperts();
      });

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
  Widget build(BuildContext context) => Scaffold(
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
                        ModelReadiness.ready =>
                          !ScreeningService.fundusDetectorAvailable
                              ? 'Screening paused — validation unavailable'
                              : useExperimentalV21
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
                                    ? ScreeningService.fundusDetectorAvailable
                                        ? 'ONNX session initialized and input/output contract verified. This status does not indicate that an image has been screened.'
                                        : 'No validated fundus detector or image-quality model is packaged. Decodable inputs are treated as uncertain, so OFFIRST will not run DR classification or record a negative result.'
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
              service: _service),
          ExpertDirectoryPage(store: _store, onChanged: _refresh),
          const _AboutPage(),
        ]),
        bottomNavigationBar: NavigationBar(
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
      );

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
              const Card(
                  child: Padding(
                      padding: EdgeInsets.all(18),
                      child: Text('Saved screening visits will appear here.')))
            else
              ...recent.take(4).map((entry) => Card(
                    child: ListTile(
                      leading: const Icon(Icons.history,
                          color: OffirstColors.cobalt),
                      title: Text(entry.patient.name),
                      subtitle: Text(
                          '${MaterialLocalizations.of(context).formatMediumDate(entry.visit.date.toLocal())} · ${entry.visit.images.length} image(s)'),
                      trailing: const Icon(Icons.chevron_right),
                      onTap: () => Navigator.of(context).push(
                        MaterialPageRoute<void>(
                          builder: (_) => VisitPage(
                              patient: entry.patient,
                              visit: entry.visit,
                              store: _store),
                        ),
                      ),
                    ),
                  )),
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
            ...rows.map((patient) => Card(
                  child: ExpansionTile(
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
                          return ListTile(
                            title: const Text('Locally associated experts'),
                            subtitle: Text(assigned.isEmpty
                                ? 'None recorded'
                                : assigned.join(', ')),
                            trailing: IconButton(
                              tooltip: 'Manage expert associations',
                              onPressed: () => _assignExperts(patient),
                              icon: const Icon(Icons.edit_outlined),
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
                )),
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
                  ? 'Active model: OFFIRST v2.1 experimental DR classifier, development-selected 0.3675977844516245 threshold. The graph includes Platt calibration. Images and records stay on this device.'
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
