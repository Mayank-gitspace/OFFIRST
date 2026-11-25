import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'offirst_theme.dart';
import 'patient_store.dart';

class ExpertDirectoryPage extends StatefulWidget {
  const ExpertDirectoryPage(
      {super.key, required this.store, required this.onChanged});
  final PatientStore store;
  final VoidCallback onChanged;

  @override
  State<ExpertDirectoryPage> createState() => _ExpertDirectoryPageState();
}

class _ExpertDirectoryPageState extends State<ExpertDirectoryPage> {
  String _query = '';
  late Future<List<ExpertRecord>> _experts = widget.store.loadExperts();
  late Future<List<PatientRecord>> _patients = widget.store.load();

  void _reload() => setState(() {
        _experts = widget.store.loadExperts();
        _patients = widget.store.load();
      });

  Future<void> _edit([ExpertRecord? original]) async {
    const fields = <(String, String)>[
      ('name', 'Full name *'),
      ('title', 'Professional title (optional)'),
      ('specialty', 'Specialty (optional)'),
      ('clinic', 'Clinic / hospital (optional)'),
      ('phone', 'Professional phone (optional)'),
      ('email', 'Professional email (optional)'),
      ('city', 'City / location (optional)'),
      ('address', 'Address (optional)'),
      ('schedule', 'Consultation days and hours (optional)'),
      ('notes', 'Notes (optional)'),
    ];
    final controllers = <String, TextEditingController>{
      for (final (key, _) in fields)
        key: TextEditingController(
            text: switch (key) {
          'name' => original?.name ?? '',
          'title' => original?.title ?? '',
          'specialty' => original?.specialty ?? '',
          'clinic' => original?.clinic ?? '',
          'phone' => original?.phone ?? '',
          'email' => original?.email ?? '',
          'city' => original?.city ?? '',
          'address' => original?.address ?? '',
          'schedule' => original?.schedule ?? '',
          _ => original?.notes ?? '',
        }),
    };
    final key = GlobalKey<FormState>();
    final save = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: Text(original == null ? 'Add expert record' : 'Edit expert'),
        content: Form(
          key: key,
          child: SizedBox(
            width: 420,
            child: SingleChildScrollView(
              child: Column(mainAxisSize: MainAxisSize.min, children: [
                if (original != null)
                  Align(
                      alignment: Alignment.centerLeft,
                      child: Padding(
                          padding: const EdgeInsets.only(bottom: 10),
                          child: Text('Expert ID  ${original.id}'))),
                for (final (field, label) in fields) ...[
                  if (field != 'name') const SizedBox(height: 8),
                  TextFormField(
                    controller: controllers[field],
                    maxLines: field == 'notes' || field == 'address' ? 2 : 1,
                    keyboardType: field == 'email'
                        ? TextInputType.emailAddress
                        : field == 'phone'
                            ? TextInputType.phone
                            : TextInputType.text,
                    decoration: InputDecoration(labelText: label),
                    validator: field == 'name'
                        ? (value) => value == null || value.trim().isEmpty
                            ? 'Enter a name.'
                            : null
                        : field == 'email'
                            ? (value) => value != null &&
                                    value.trim().isNotEmpty &&
                                    !value.contains('@')
                                ? 'Enter a valid email address.'
                                : null
                            : null,
                  ),
                ],
                const SizedBox(height: 12),
                const Text(
                    'Details are entered locally and are not professionally verified.'),
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
                if (key.currentState!.validate()) {
                  Navigator.pop(dialogContext, true);
                }
              },
              child: const Text('Save')),
        ],
      ),
    );
    if (save != true || !mounted) return;
    String? value(String name) {
      final text = controllers[name]!.text.trim();
      return text.isEmpty ? null : text;
    }

    final record = ExpertRecord(
      id: original?.id ?? recordId('EXP'),
      name: value('name')!,
      updatedAt: DateTime.now().toUtc(),
      title: value('title'),
      specialty: value('specialty'),
      clinic: value('clinic'),
      phone: value('phone'),
      email: value('email'),
      city: value('city'),
      address: value('address'),
      schedule: value('schedule'),
      notes: value('notes'),
    );
    try {
      if (original == null) {
        await widget.store.addExpert(record);
      } else {
        await widget.store.updateExpert(record);
      }
      if (mounted) {
        _reload();
        widget.onChanged();
      }
    } catch (_) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Could not save the expert record.')));
      }
    }
  }

  Future<void> _delete(ExpertRecord expert) async {
    final patients = await _patients;
    if (!mounted) return;
    final count = patients.where((p) => p.expertIds.contains(expert.id)).length;
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: const Text('Delete expert record?'),
        content: Text(
            'Delete ${expert.name} (${expert.id}) and remove its local association from $count patient record(s)?'),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(dialogContext, false),
              child: const Text('Cancel')),
          FilledButton.tonal(
              onPressed: () => Navigator.pop(dialogContext, true),
              child: const Text('Delete')),
        ],
      ),
    );
    if (confirmed != true || !mounted) return;
    try {
      await widget.store.deleteExpert(expert.id);
      if (mounted) {
        _reload();
        widget.onChanged();
      }
    } catch (_) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
            content: Text('Could not delete the expert record.')));
      }
    }
  }

  @override
  Widget build(BuildContext context) => FutureBuilder<List<ExpertRecord>>(
        future: _experts,
        builder: (context, expertSnapshot) {
          if (expertSnapshot.connectionState != ConnectionState.done) {
            return const Center(child: CircularProgressIndicator());
          }
          if (expertSnapshot.hasError) {
            return const Center(child: Text('Could not load expert records.'));
          }
          final query = _query.toLowerCase().trim();
          final experts = (expertSnapshot.data ?? [])
              .where((expert) =>
                  expert.name.toLowerCase().contains(query) ||
                  expert.id.toLowerCase().contains(query) ||
                  (expert.specialty ?? '').toLowerCase().contains(query))
              .toList();
          return ListView(padding: const EdgeInsets.all(16), children: [
            Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
              Text('Experts', style: Theme.of(context).textTheme.headlineSmall),
              IconButton(
                  tooltip: 'Add expert record',
                  onPressed: _edit,
                  icon: const Icon(Icons.person_add_alt_outlined)),
            ]),
            const SizedBox(height: 10),
            TextField(
                onChanged: (value) => setState(() => _query = value),
                decoration: const InputDecoration(
                    prefixIcon: Icon(Icons.search),
                    hintText: 'Search name, ID or specialty')),
            const SizedBox(height: 12),
            const Card(
                child: Padding(
                    padding: EdgeInsets.all(14),
                    child: Text(
                        'Locally entered professional details are unverified. Recorded schedules are not live availability; confirm directly.'))),
            if (experts.isEmpty)
              const Padding(
                  padding: EdgeInsets.all(24),
                  child: Text('No local expert records. Add one to begin.')),
            ...experts.map((expert) => Card(
                  child: ExpansionTile(
                    leading: const Icon(Icons.badge_outlined,
                        color: OffirstColors.cobalt),
                    title: Text(expert.name),
                    subtitle: Text([
                      if (expert.title != null) expert.title!,
                      if (expert.specialty != null) expert.specialty!,
                    ].join(' · ')),
                    trailing: PopupMenuButton<String>(
                      onSelected: (value) =>
                          value == 'edit' ? _edit(expert) : _delete(expert),
                      itemBuilder: (_) => const [
                        PopupMenuItem(value: 'edit', child: Text('Edit')),
                        PopupMenuItem(value: 'delete', child: Text('Delete')),
                      ],
                    ),
                    children: [
                      ListTile(
                        title: const Text('Expert ID'),
                        subtitle: SelectableText(expert.id),
                        trailing: IconButton(
                          tooltip: 'Copy expert ID',
                          icon: const Icon(Icons.copy_outlined),
                          onPressed: () async {
                            await Clipboard.setData(
                                ClipboardData(text: expert.id));
                            if (context.mounted) {
                              ScaffoldMessenger.of(context).showSnackBar(
                                  const SnackBar(
                                      content: Text('Expert ID copied.')));
                            }
                          },
                        ),
                      ),
                      if (expert.clinic != null)
                        ListTile(
                            title: const Text('Clinic / hospital'),
                            subtitle: Text(expert.clinic!)),
                      if (expert.phone != null)
                        ListTile(
                            title: const Text('Phone'),
                            subtitle: SelectableText(expert.phone!)),
                      if (expert.email != null)
                        ListTile(
                            title: const Text('Email'),
                            subtitle: SelectableText(expert.email!)),
                      if (expert.city != null)
                        ListTile(
                            title: const Text('Location'),
                            subtitle: Text(expert.city!)),
                      if (expert.address != null)
                        ListTile(
                            title: const Text('Address'),
                            subtitle: Text(expert.address!)),
                      if (expert.schedule != null)
                        ListTile(
                            title: const Text(
                                'Locally recorded availability — confirm directly'),
                            subtitle: Text(expert.schedule!)),
                      ListTile(
                          title: const Text('Last updated'),
                          subtitle: Text(MaterialLocalizations.of(context)
                              .formatMediumDate(expert.updatedAt.toLocal()))),
                      FutureBuilder<List<PatientRecord>>(
                        future: _patients,
                        builder: (context, patientSnapshot) {
                          final names = (patientSnapshot.data ?? [])
                              .where((p) => p.expertIds.contains(expert.id))
                              .map((p) => p.name)
                              .toList();
                          return ListTile(
                              title: const Text('Associated patient records'),
                              subtitle: Text(names.isEmpty
                                  ? 'None recorded'
                                  : names.join(', ')));
                        },
                      ),
                      const Padding(
                          padding: EdgeInsets.fromLTRB(16, 0, 16, 12),
                          child: Text(
                              'An association does not mean this expert reviewed or approved a screening result.')),
                    ],
                  ),
                )),
          ]);
        },
      );
}
